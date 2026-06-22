from __future__ import annotations

import math
import queue
import sys
import time

import numpy as np
import pygame
import carla

from .config import CONFIG
from .modes.driving_mode import DrivingMode
from .input_handler import (
    InputType, MouseInput, WheelInput, ControlState, create_input,
)


class ClientDriver:
    """Sets up CARLA + pygame and runs the main driving loop."""

    def __init__(
        self,
        mode: DrivingMode,
        *,
        carla_host: str = "127.0.0.1",
        carla_port: int = 2000,
        vehicle_type: str = "vehicle.tesla.model3",
        town: str | None = None,
        traffic: int = 0,
        resolution: tuple[int, int] = (1280, 720),
        input_type: InputType = InputType.MOUSE,
        debug_axes: bool = False,
    ) -> None:
        self.mode: DrivingMode = mode
        self.dw: int
        self.dh: int
        self.dw, self.dh = resolution

        pygame.init()
        pygame.font.init()
        self.display: pygame.Surface = pygame.display.set_mode(
            (self.dw, self.dh), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("CARLA Drowsiness Detection")
        self.font: pygame.font.Font = pygame.font.SysFont("monospace", 18)
        self.font_big: pygame.font.Font = pygame.font.SysFont(
            "monospace", 48, bold=True)
        self.clock: pygame.time.Clock = pygame.time.Clock()

        self.input: MouseInput | WheelInput = create_input(
            input_type, debug_axes=debug_axes)
        print(f"Input device: {self.input.device_name}")

        # Mouse grab (only relevant for MouseInput)
        self.grabbed: bool = True
        if isinstance(self.input, MouseInput):
            pygame.event.set_grab(True)
            pygame.mouse.set_visible(False)
            pygame.mouse.get_rel()
        else:
            # Wheel doesn't need mouse capture
            pygame.event.set_grab(False)
            pygame.mouse.set_visible(True)

        self.client: carla.Client = carla.Client(carla_host, carla_port)
        self.client.set_timeout(20.0)

        if town is not None:
            current_map: str = self.client.get_world().get_map().name
            if town.lower() not in current_map.lower():
                print(f"Loading {town}...")
                self.world: carla.World = self.client.load_world(town)
            else:
                print(f"Map {town} already loaded.")
                self.world = self.client.get_world()
        else:
            self.world = self.client.get_world()

        self.world_map: carla.Map = self.world.get_map()
        self.original_settings: carla.WorldSettings = self.world.get_settings()

        settings: carla.WorldSettings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = CONFIG.sim.fixed_delta
        self.world.apply_settings(settings)

        self.traffic_manager: carla.TrafficManager = (
            self.client.get_trafficmanager())
        self.tm_port: int = self.traffic_manager.get_port()
        self.traffic_manager.set_synchronous_mode(True)

        self.actors: list[carla.Actor] = []
        self.image_queue: queue.Queue[carla.Image] = queue.Queue()
        self.spawn_points: list[carla.Transform] = (
            self.world_map.get_spawn_points())
        if not self.spawn_points:
            sys.exit("Map has no spawn points.")

        bp_lib: carla.BlueprintLibrary = self.world.get_blueprint_library()

        vehicle_bp: carla.ActorBlueprint = bp_lib.filter(vehicle_type)[0]
        spawn: carla.Transform = self.spawn_points[
            np.random.randint(len(self.spawn_points))]
        self.vehicle: carla.Vehicle = self.world.spawn_actor(
            vehicle_bp, spawn)
        self.actors.append(self.vehicle)

        if traffic > 0:
            npc_bps = bp_lib.filter("vehicle.*")
            free: list[carla.Transform] = [
                p for p in self.spawn_points
                if p.location != spawn.location
            ]
            np.random.shuffle(free)
            spawned: int = 0
            for p in free[:traffic]:
                npc: carla.Actor | None = self.world.try_spawn_actor(
                    npc_bps[np.random.randint(len(npc_bps))], p)
                if npc is not None:
                    npc.set_autopilot(True, self.tm_port)
                    self.actors.append(npc)
                    spawned += 1
            print(f"Spawned {spawned} NPC vehicles.")

        cam_bp: carla.ActorBlueprint = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(self.dw))
        cam_bp.set_attribute("image_size_y", str(self.dh))
        cam_bp.set_attribute("fov", "90")
        cam_tf: carla.Transform = carla.Transform(
            carla.Location(x=-5.5, z=2.8),
            carla.Rotation(pitch=-12.0))
        self.camera: carla.Actor = self.world.spawn_actor(
            cam_bp, cam_tf, attach_to=self.vehicle)
        self.actors.append(self.camera)
        self.camera.listen(self.image_queue.put)

        self.steer: float = 0.0
        self.steer_raw: float = 0.0
        self.throttle: float = 0.0
        self.brake: float = 0.0
        self.reverse: bool = False
        self.speed_kmh: float = 0.0
        self.tick_counter: int = 0
        self.start_wall: float = time.time()

    def run(self) -> None:
        self.mode.on_setup(self)
        running: bool = True
        try:
            while running:
                self.world.tick()
                snapshot: carla.WorldSnapshot = self.world.get_snapshot()
                self.tick_counter += 1

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_r:
                            self.reverse = not self.reverse
                        elif event.key == pygame.K_g:
                            self._toggle_grab()
                        elif event.key == pygame.K_TAB:
                            self._respawn()
                        else:
                            self.mode.on_key(event.key, self)
                    else:
                        self.input.on_event(event)

                if isinstance(self.input, WheelInput) and self.input.reverse_toggled:
                    self.reverse = not self.reverse

                self._update_controls()
                self._update_vehicle_state()
                self.mode.on_tick(self, snapshot)
                self._render()
                self.clock.tick(CONFIG.sim.hz)
        finally:
            self._shutdown()

    def _toggle_grab(self) -> None:
        if isinstance(self.input, MouseInput):
            self.grabbed = not self.grabbed
            self.input.grabbed = self.grabbed
            pygame.event.set_grab(self.grabbed)
            pygame.mouse.set_visible(not self.grabbed)
            pygame.mouse.get_rel()

    def _update_controls(self) -> None:
        ctrl: ControlState = self.input.read()
        self.steer = ctrl.steer
        self.steer_raw = ctrl.steer_raw
        self.throttle = ctrl.throttle
        self.brake = ctrl.brake

        control: carla.VehicleControl = carla.VehicleControl(
            throttle=float(self.throttle),
            steer=float(self.steer),
            brake=float(self.brake),
            reverse=self.reverse,
        )
        if self.reverse:
            control.manual_gear_shift = True
            control.gear = -1
        else:
            control.manual_gear_shift = False
        self.vehicle.apply_control(control)

    def _update_vehicle_state(self) -> None:
        vel: carla.Vector3D = self.vehicle.get_velocity()
        self.speed_kmh = 3.6 * math.sqrt(
            vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

    def _respawn(self) -> None:
        sp: carla.Transform = self.spawn_points[
            np.random.randint(len(self.spawn_points))]
        self.vehicle.set_transform(sp)
        self.vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
        self.steer = 0.0
        if isinstance(self.input, MouseInput):
            self.input.reset_steer()
        if hasattr(self.mode, "on_respawn"):
            self.mode.on_respawn()

    def _render(self) -> None:
        try:
            image: carla.Image = self.image_queue.get(timeout=2.0)
            arr: np.ndarray = np.frombuffer(image.raw_data, dtype=np.uint8)
            arr = arr.reshape((image.height, image.width, 4))[:, :, :3][:, :, ::-1]
            surface: pygame.Surface = pygame.surfarray.make_surface(
                arr.swapaxes(0, 1))
            self.display.blit(surface, (0, 0))
        except queue.Empty:
            pass

        warning_type: str | None
        warning_type, _ = self.mode.get_warning_state()
        self._draw_warning_overlay(warning_type)
        self._draw_hud()
        pygame.display.flip()

    def _draw_warning_overlay(self, warning_type: str | None) -> None:
        dw: int = self.dw
        dh: int = self.dh

        if warning_type == "INACTIVITY":
            overlay: pygame.Surface = pygame.Surface(
                (dw, dh), pygame.SRCALPHA)
            for rect in [(0, 0, dw, 10), (0, dh - 10, dw, 10),
                         (0, 0, 10, dh), (dw - 10, 0, 10, dh)]:
                pygame.draw.rect(overlay, (255, 200, 0, 160), rect)
            self.display.blit(overlay, (0, 0))
            # txt: pygame.Surface = self.font_big.render(
            #     "WAKE UP!", True, (255, 200, 0))
            # self.display.blit(
            #     txt, txt.get_rect(center=(dw // 2, dh // 2 - 40)))

        elif warning_type == "DROWSY":
            overlay = pygame.Surface((dw, dh), pygame.SRCALPHA)
            for rect in [(0, 0, dw, 12), (0, dh - 12, dw, 12),
                         (0, 0, 12, dh), (dw - 12, 0, 12, dh)]:
                pygame.draw.rect(overlay, (255, 0, 0, 140), rect)
            self.display.blit(overlay, (0, 0))
            # txt = self.font_big.render(
            #     "!! DROWSY !!", True, (255, 40, 40))
            # self.display.blit(
            #     txt, txt.get_rect(center=(dw // 2, dh // 2 - 40)))

    def _draw_hud(self) -> None:
        elapsed: float = time.time() - self.start_wall
        input_tag: str = self.input.input_type_str
        base_lines: list[str] = [
            f"t={elapsed:5.0f}s  speed={self.speed_kmh:5.1f} km/h"
            f"  steer={self.steer:+.2f}  [{input_tag}]",
        ]
        mode_lines: list[str] = self.mode.get_hud_lines(self)
        controls: str = (
            "W/S=pedals R=reverse G=mouse TAB=respawn ESC=quit"
            if isinstance(self.input, MouseInput)
            else "pedals=gas/brake R=reverse TAB=respawn ESC=quit"
        )
        all_lines: list[str] = base_lines + mode_lines + [controls]

        for i, line in enumerate(all_lines):
            txt: pygame.Surface = self.font.render(
                line, True, (255, 255, 80))
            self.display.blit(txt, (12, 10 + i * 22))

    def _shutdown(self) -> None:
        print("\nShutting down...")
        self.mode.on_cleanup()
        for actor in self.actors:
            try:
                actor.destroy()
            except Exception:
                pass
        try:
            self.world.apply_settings(self.original_settings)
        except Exception:
            pass
        try:
            self.traffic_manager.set_synchronous_mode(False)
        except Exception:
            pass
        pygame.quit()
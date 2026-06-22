from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import pygame

from .config import CONFIG


class InputType(Enum):
    MOUSE = "mouse"
    WHEEL = "wheel"


@dataclass(frozen=True, slots=True)
class ControlState:
    """One tick's worth of driver input."""
    steer: float      # [-1, 1]
    steer_raw: float   # raw device value before any curve (for logging)
    throttle: float    # [0, 1]
    brake: float       # [0, 1]
    input_type: str    # "mouse_kb" or "wheel"


class InputHandler(Protocol):
    """Interface for input devices."""
    def read(self) -> ControlState: ...
    def on_event(self, event: pygame.event.Event) -> None: ...
    @property
    def device_name(self) -> str: ...
    @property
    def input_type_str(self) -> str: ...

class MouseInput:
    """Continuous mouse steering + keyboard pedals."""

    def __init__(self) -> None:
        self._steer: float = 0.0
        self._grabbed: bool = True
        pygame.mouse.get_rel()  # reset baseline

    @property
    def device_name(self) -> str:
        return "Mouse + Keyboard"

    @property
    def input_type_str(self) -> str:
        return "mouse_kb"

    @property
    def grabbed(self) -> bool:
        return self._grabbed

    @grabbed.setter
    def grabbed(self, value: bool) -> None:
        self._grabbed = value

    def reset_steer(self) -> None:
        self._steer = 0.0

    def read(self) -> ControlState:
        cfg = CONFIG.steering

        if self._grabbed:
            mdx: int
            mdx, _ = pygame.mouse.get_rel()
            self._steer += mdx * cfg.sensitivity
        self._steer *= cfg.autocenter
        self._steer = max(-1.0, min(1.0, self._steer))

        keys: pygame.key.ScancodeWrapper = pygame.key.get_pressed()
        throttle: float = 1.0 if keys[pygame.K_w] else 0.0
        brake: float = 1.0 if (keys[pygame.K_s] or keys[pygame.K_SPACE]) else 0.0

        return ControlState(
            steer=self._steer,
            steer_raw=self._steer,
            throttle=throttle,
            brake=brake,
            input_type="mouse_kb",
        )

    def on_event(self, event: pygame.event.Event) -> None:
        pass  

class WheelInput:
    """Physical steering wheel with pedals via pygame joystick."""

    def __init__(self, debug_axes: bool = False) -> None:
        self._debug: bool = debug_axes
        self._reverse_pressed: bool = False

        pygame.joystick.init()
        count: int = pygame.joystick.get_count()
        if count == 0:
            raise RuntimeError(
                "No steering wheel / joystick detected. "
                "Plug it in and restart, or use --input mouse."
            )

        self._joy: pygame.joystick.Joystick = pygame.joystick.Joystick(0)
        self._joy.init()
        print(
            f"Wheel detected: {self._joy.get_name()}  "
            f"({self._joy.get_numaxes()} axes, "
            f"{self._joy.get_numbuttons()} buttons)"
        )

    @property
    def device_name(self) -> str:
        return self._joy.get_name()

    @property
    def input_type_str(self) -> str:
        return "wheel"

    def read(self) -> ControlState:
        cfg = CONFIG.wheel
        n_axes: int = self._joy.get_numaxes()

        # Read raw axes (safe fallback if axis doesn't exist)
        def axis(idx: int, default: float = 0.0) -> float:
            return self._joy.get_axis(idx) if idx < n_axes else default

        raw_steer: float = axis(cfg.steer_axis)
        raw_throttle: float = axis(cfg.throttle_axis, 1.0)
        raw_brake: float = axis(cfg.brake_axis, 1.0)

        if self._debug:
            axes_str: str = "  ".join(
                f"{i}:{self._joy.get_axis(i):+.3f}" for i in range(n_axes))
            btns_str: str = "".join(
                str(self._joy.get_button(i))
                for i in range(self._joy.get_numbuttons()))
            print(f"AXES: {axes_str}  BTNS: {btns_str}")

        steer: float = cfg.steer_nonlinearity * math.tan(1.1 * raw_steer)
        steer = max(-1.0, min(1.0, steer))

        throttle: float = self._pedal_map(raw_throttle, cfg.pedal_deadzone)
        brake: float = self._pedal_map(raw_brake, cfg.pedal_deadzone)

        return ControlState(
            steer=steer,
            steer_raw=raw_steer,
            throttle=throttle,
            brake=brake,
            input_type="wheel",
        )

    def on_event(self, event: pygame.event.Event) -> None:
        """Track reverse button press (toggle)."""
        cfg = CONFIG.wheel
        if event.type == pygame.JOYBUTTONDOWN:
            if event.button == cfg.reverse_button:
                self._reverse_pressed = True

    @property
    def reverse_toggled(self) -> bool:
        """Check and consume a reverse toggle."""
        if self._reverse_pressed:
            self._reverse_pressed = False
            return True
        return False

    @staticmethod
    def _pedal_map(raw: float, deadzone: float) -> float:
        """Map pedal axis from [+1 (rest) .. -1 (pressed)] to [0 .. 1]"""
        cmd: float = 1.6 + (2.05 * math.log10(
            max(0.001, -0.7 * raw + 1.4)) - 1.2) / 0.92
        cmd = max(0.0, min(1.0, cmd))
        return cmd if cmd > deadzone else 0.0


def create_input(
    input_type: InputType,
    debug_axes: bool = False,
) -> MouseInput | WheelInput:
    """Create the appropriate input handler."""
    if input_type == InputType.WHEEL:
        return WheelInput(debug_axes=debug_axes)
    return MouseInput()
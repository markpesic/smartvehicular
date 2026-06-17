from __future__ import annotations

import csv
import datetime
import os
import time
from io import TextIOWrapper
from typing import TYPE_CHECKING

from ..config import CONFIG
from ..utils.lane_metrics import compute_lane_metrics

if TYPE_CHECKING:
    import carla
    from ..client_driver import ClientDriver
    from ..modes.driving_mode import WarningState

CSV_HEADER: list[str] = [
    "wall_time", "sim_time", "frame", "dt",
    "steer_raw", "steer_cmd", "throttle", "brake", "reverse",
    "speed_kmh", "loc_x", "loc_y", "loc_z", "yaw",
    "cte", "heading_err_deg", "lane_width", "curvature", "on_junction",
    "kss", "sleep_hours", "session_type", "input_device", "event_flag",
]


class LoggerMode:
    """Plugs into ``ClientDriver`` to log steering data for offline analysis."""

    def __init__(
        self,
        session_type: str,
        sleep_hours: float,
        outdir: str = "logs",
    ) -> None:
        self.session_type: str = session_type
        self.sleep_hours: float = sleep_hours
        self.outdir: str = outdir

        self.current_kss: int = 1 if session_type == "alert" else 5
        self.event_flag: int = 0
        self._csv_file: TextIOWrapper | None = None
        self._writer: csv.writer | None = None
        self.csv_path: str | None = None
        self._last_kss_reminder: float = 0.0

    def on_setup(self, driver: ClientDriver) -> None:
        os.makedirs(self.outdir, exist_ok=True)
        stamp: str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(
            self.outdir, f"drive_{self.session_type}_{stamp}.csv")
        self._csv_file = open(self.csv_path, "w", newline="")
        self._writer = csv.writer(self._csv_file)
        self._writer.writerow(CSV_HEADER)
        self._last_kss_reminder = time.time()
        print(f"Logging to: {self.csv_path}")

    def on_key(self, key: int, driver: ClientDriver) -> bool:
        import pygame

        if pygame.K_1 <= key <= pygame.K_9:
            self.current_kss = key - pygame.K_0
            print(f"  KSS set to {self.current_kss}")
            return True
        if key == pygame.K_m:
            self.event_flag = 1
            print("  >> event marker")
            return True
        return False

    def on_tick(self, driver: ClientDriver, snapshot: carla.WorldSnapshot) -> None:
        assert self._writer is not None

        metrics = compute_lane_metrics(driver.world.get_map(), driver.vehicle)
        tf: carla.Transform = driver.vehicle.get_transform()
        dt: float = CONFIG.sim.fixed_delta

        self._writer.writerow([
            f"{time.time():.6f}",
            f"{snapshot.timestamp.elapsed_seconds:.4f}",
            snapshot.frame,
            f"{dt:.4f}",
            f"{driver.steer:.5f}",
            f"{driver.steer:.5f}",
            f"{driver.throttle:.4f}",
            f"{driver.brake:.4f}",
            int(driver.reverse),
            f"{driver.speed_kmh:.3f}",
            f"{tf.location.x:.3f}",
            f"{tf.location.y:.3f}",
            f"{tf.location.z:.3f}",
            f"{tf.rotation.yaw:.3f}",
            f"{metrics.cte:.4f}",
            f"{metrics.heading_err_deg:.3f}",
            f"{metrics.lane_width:.3f}",
            f"{metrics.curvature:.4f}",
            metrics.on_junction,
            self.current_kss,
            self.sleep_hours,
            self.session_type,
            "mouse_kb",
            self.event_flag,
        ])
        self.event_flag = 0

        if time.time() - self._last_kss_reminder > 180:
            print("  (reminder: update your KSS rating)")
            self._last_kss_reminder = time.time()

    def get_hud_lines(self, driver: ClientDriver) -> list[str]:
        return [
            f"REC {os.path.basename(self.csv_path or '')}",
            f"KSS {self.current_kss}   session {self.session_type}"
            f"   sleep {self.sleep_hours}h",
        ]

    def get_warning_state(self) -> WarningState:
        return None, 0

    def on_cleanup(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
            print(f"Saved: {self.csv_path}")

from __future__ import annotations

import collections
import json
import pickle
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from sklearn.pipeline import Pipeline

from ..config import CONFIG
from ..feature_extraction import extract_features
from ..utils.mqtt_warner import MQTTWarner

if TYPE_CHECKING:
    import carla
    from client_driver import ClientDriver
    from modes import WarningState


class InferenceMode:
    """Plugs into ClientDriver for live drowsiness detection."""

    def __init__(
        self,
        model_path: str = "drowsy_model.pkl",
        meta_path: str = "drowsy_model_meta.json",
        mqtt_host: str = "127.0.0.1",
        mqtt_port: int = 1883,
        predict_every_s: float = 2,
        smooth_n: int = 2,
    ) -> None:
        # Load trained model + metadata
        with open(model_path, "rb") as f:
            self.model: Pipeline = pickle.load(f)
        with open(meta_path) as f:
            meta: dict[str, Any] = json.load(f)

        self.feat_cols: list[str] = meta["feature_cols"]
        self.alpha: float = meta.get("alpha", 0.01)
        self.fs: int = int(meta.get("fs", 20))
        self.window_s: float = float(meta.get("window_s", 15.0))
        self.win_samples: int = int(round(self.window_s * self.fs))
        self.pos_idx: int = int(list(self.model.classes_).index(1))

        self.predict_every_s: float = predict_every_s
        self.smooth_n: int = smooth_n

        self.warner: MQTTWarner = MQTTWarner(mqtt_host, mqtt_port)

        self.steer_buffer: collections.deque[float] = collections.deque(
            maxlen=self.win_samples)
        self.predict_interval: int = 0
        self.recent_preds: collections.deque[int] = collections.deque(
            maxlen=smooth_n)

        self.current_state: str = "ALERT"
        self.current_prob: float = 0.0
        self.consecutive_drowsy: int = 0
        self.warning_flash: int = 0
        self.inactivity_flash: int = 0
        self.inactive_ticks: int = 0
        self._prev_steer: float = 0.0

        print(
            f"Model loaded: {len(self.feat_cols)} features, "
            f"window={self.window_s}s, alpha={self.alpha:.5f}"
        )

    def on_setup(self, driver: ClientDriver) -> None:
        hz: int = CONFIG.sim.hz
        self.predict_interval = int(round(self.predict_every_s * hz))
        print(
            f"Predictions every {self.predict_every_s}s, "
            f"smoothed over {self.smooth_n} windows."
        )
        print("Subscribe:  mosquitto_sub -t 'carla/drowsiness/#'\n")

    def on_key(self, key: int, driver: ClientDriver) -> bool:
        return False  

    def on_tick(self, driver: ClientDriver, snapshot: carla.WorldSnapshot) -> None:
        steer: float = driver.steer
        speed_kmh: float = driver.speed_kmh
        tick: int = driver.tick_counter
        hz: int = CONFIG.sim.hz
        inact_cfg = CONFIG.inactivity

        self.steer_buffer.append(steer)

        # steering inactivity
        steer_delta: float = abs(steer - self._prev_steer)
        self._prev_steer = steer

        if (steer_delta < inact_cfg.steer_thresh
                and speed_kmh > inact_cfg.min_speed_kmh):
            self.inactive_ticks += 1
        else:
            self.inactive_ticks = 0

        inactive_seconds: float = self.inactive_ticks / hz
        if inactive_seconds >= inact_cfg.time_s:
            if self.inactive_ticks % hz == 0:
                self.warner.publish_inactivity(inactive_seconds, speed_kmh)
                print(
                    f"  >> INACTIVITY  {inactive_seconds:.1f}s still  "
                    f"speed={speed_kmh:.0f} km/h"
                )
            self.inactivity_flash = max(self.inactivity_flash, hz // 2)

        # ML prediction
        if (tick % self.predict_interval == 0
                and len(self.steer_buffer) >= self.win_samples):
            buf: npt.NDArray[np.floating] = np.array(
                self.steer_buffer, dtype=float)
            feats: dict[str, float] = extract_features(
                buf, self.fs, self.alpha)
            x: npt.NDArray[np.floating] = np.array(
                [[feats[c] for c in self.feat_cols]])
            prob: float = float(
                self.model.predict_proba(x)[0, self.pos_idx])
            pred: int = int(prob >= 0.5)
            self.recent_preds.append(pred)

            if prob >= 0.80:
                new_state = "DROWSY"
            elif len(self.recent_preds) >= self.smooth_n:
                vote: bool = sum(self.recent_preds) > len(self.recent_preds) / 2.0
                new_state = "DROWSY" if vote else "ALERT"
            else:
                new_state = "ALERT"

            if new_state == "DROWSY":
                self.consecutive_drowsy += 1
            else:
                self.consecutive_drowsy = 0

            self.current_prob = prob
            self.current_state = new_state

            self.warner.publish_prediction(
                self.current_state, self.current_prob, self.consecutive_drowsy)

            if self.current_state == "DROWSY":
                self.warning_flash = self.predict_interval
                print(
                    f"  !! DROWSY  prob={prob:.2f}  "
                    f"consecutive={self.consecutive_drowsy}"
                )

    def get_hud_lines(self, driver: ClientDriver) -> list[str]:
        return [
            f"STATE: {self.current_state}  P(drowsy)={self.current_prob:.2f}"
            f"  consecutive={self.consecutive_drowsy}",
            f"buffer: {len(self.steer_buffer)}/{self.win_samples}  "
            f"MQTT: {'connected' if self.warner.connected else 'disconnected'}",
        ]

    def get_warning_state(self) -> WarningState:
        if self.warning_flash > 0:
            self.warning_flash -= 1
            return "DROWSY", self.warning_flash
        if self.inactivity_flash > 0:
            self.inactivity_flash -= 1
            return "INACTIVITY", self.inactivity_flash
        return None, 0

    def on_respawn(self) -> None:
        """Reset buffers and state on TAB respawn."""
        self.steer_buffer.clear()
        self.recent_preds.clear()
        self.current_state = "ALERT"
        self.warning_flash = 0
        self.inactivity_flash = 0
        self.inactive_ticks = 0

    def on_cleanup(self) -> None:
        self.warner.close()

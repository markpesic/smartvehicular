from __future__ import annotations

import json
import time

import paho.mqtt.client as mqtt


class MQTTWarner:
    """Publishes drowsiness state, warnings, and heartbeats over MQTT."""

    TOPIC_STATE: str = "carla/drowsiness/state"
    TOPIC_WARNING: str = "carla/drowsiness/warning"
    TOPIC_INACTIVITY: str = "carla/drowsiness/inactivity"
    TOPIC_HEARTBEAT: str = "carla/drowsiness/heartbeat"

    def __init__(self, host: str, port: int) -> None:
        self.client: mqtt.Client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.host: str = host
        self.port: int = port
        self.connected: bool = False
        self.last_state: str | None = None
        self._connect()

    def _connect(self) -> None:
        try:
            self.client.connect(self.host, self.port, keepalive=60)
            self.client.loop_start()
            self.connected = True
            print(f"MQTT connected to {self.host}:{self.port}")
        except Exception as exc:
            print(f"MQTT connection failed ({exc}); HUD-only mode.")
            self.connected = False

    def _publish_state(self, state: str) -> None:
        if self.connected and state != self.last_state:
            self.client.publish(self.TOPIC_STATE, state, qos=1, retain=True)
            self.last_state = state

    def publish_prediction(
        self,
        state: str,
        prob: float,
        consecutive_drowsy: int,
    ) -> None:
        """Publish heartbeat + optional warning on every ML prediction."""
        if not self.connected:
            return
        ts: float = time.time()
        self.client.publish(self.TOPIC_HEARTBEAT, json.dumps({
            "state": state,
            "drowsy_prob": round(prob, 3),
            "consecutive_drowsy": consecutive_drowsy,
            "timestamp": ts,
        }), qos=0)
        self._publish_state(state)
        if state == "DROWSY":
            self.client.publish(self.TOPIC_WARNING, json.dumps({
                "drowsy_prob": round(prob, 3),
                "consecutive_drowsy": consecutive_drowsy,
                "timestamp": ts,
            }), qos=1)

    def publish_inactivity(
        self,
        inactive_seconds: float,
        speed_kmh: float,
    ) -> None:
        """Publish a Tier-1 steering-inactivity early warning."""
        if not self.connected:
            return
        self._publish_state("INACTIVITY")
        self.client.publish(self.TOPIC_INACTIVITY, json.dumps({
            "inactive_seconds": round(inactive_seconds, 1),
            "speed_kmh": round(speed_kmh, 1),
            "timestamp": time.time(),
        }), qos=1)

    def close(self) -> None:
        """Gracefully disconnect."""
        if self.connected:
            self.client.publish(
                self.TOPIC_STATE, "OFFLINE", qos=1, retain=True)
            self.client.loop_stop()
            self.client.disconnect()

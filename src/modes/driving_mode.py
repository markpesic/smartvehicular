from __future__ import annotations
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import carla
    from client_driver import ClientDriver


WarningState = tuple[str | None, int]
"""(warning_type, ticks_remaining) — None means no active warning."""


@runtime_checkable
class DrivingMode(Protocol):
    """Interface that every driving mode must implement.

    ``ClientDriver`` calls these hooks at well-defined points in the
    driving loop.  Modes must not call pygame or CARLA directly —
    they receive the driver reference for reading shared state and
    return data that the driver renders.
    """

    def on_setup(self, driver: ClientDriver) -> None:
        """Called once after CARLA is connected and the vehicle is spawned."""
        ...

    def on_key(self, key: int, driver: ClientDriver) -> bool:
        """Handle a mode-specific keypress.  Return True if consumed."""
        ...

    def on_tick(self, driver: ClientDriver, snapshot: carla.WorldSnapshot) -> None:
        """Called every 20 Hz tick after steering is applied."""
        ...

    def get_hud_lines(self, driver: ClientDriver) -> list[str]:
        """Return 1-3 text lines for the mode-specific HUD section."""
        ...

    def get_warning_state(self) -> WarningState:
        """Return the current warning overlay type and ticks remaining."""
        ...

    def on_cleanup(self) -> None:
        """Called on shutdown.  Release resources (files, MQTT, etc.)."""
        ...

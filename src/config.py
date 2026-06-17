from __future__ import annotations

from pydantic import BaseModel, Field


class SimConfig(BaseModel):
    """CARLA simulation timing."""
    hz: int = Field(default=20, description="Logging / control rate (Hz)")

    @property
    def fixed_delta(self) -> float:
        return 1.0 / self.hz


class SteeringConfig(BaseModel):
    """Mouse steering feel — tune once, keep fixed across sessions."""
    sensitivity: float = Field(
        default=0.0015,
        description="Steer units added per pixel of mouse movement",
    )
    autocenter: float = Field(
        default=0.98,
        ge=0.0, le=1.0,
        description="Per-tick spring toward straight (1.0 = none)",
    )


class InactivityConfig(BaseModel):
    """Tier-1 steering-inactivity early-warning thresholds."""
    steer_thresh: float = Field(
        default=0.003,
        description="Max |Δsteering| per tick to count as 'still'",
    )
    time_s: float = Field(
        default=3.0, gt=0,
        description="Seconds of stillness before early warning",
    )
    min_speed_kmh: float = Field(
        default=15.0, ge=0,
        description="Only warn if the car is moving above this speed",
    )


class FeatureConfig(BaseModel):
    """Constants for the 11 steering-only features."""
    reversal_gap: float = Field(default=0.02)
    hold_vel_thresh: float = Field(default=0.005)
    large_corr_thresh: float = Field(default=0.10)
    lf_band: tuple[float, float] = Field(default=(0.0, 0.2))
    hf_band: tuple[float, float] = Field(default=(0.2, 2.0))
    entropy_bins_k: tuple[float, float, float, float] = Field(
        default=(5.0, 2.5, 1.0, 0.5),
        description="Nakayama 9-bin edges as multiples of alpha",
    )


class AppConfig(BaseModel):
    """Top-level config aggregating all sub-configs."""
    sim: SimConfig = Field(default_factory=SimConfig)
    steering: SteeringConfig = Field(default_factory=SteeringConfig)
    inactivity: InactivityConfig = Field(default_factory=InactivityConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)


class ExtractionMeta(BaseModel):
    """Metadata saved alongside features.csv for reproducibility."""
    alpha: float = Field(description="Steering-entropy scale factor")
    fs: int = Field(description="Sampling rate (Hz)")
    window_s: float = Field(description="Window length (seconds)")
    overlap: float = Field(description="Window overlap fraction")


class SessionResult(BaseModel):
    """Summary of one session's extraction."""
    session: str
    label: str
    fs: int
    n_windows: int

    class Config:
        arbitrary_types_allowed = True


class ModelMeta(BaseModel):
    """Metadata saved alongside the pickled model for live inference."""
    feature_cols: list[str]
    pos_label: str = "drowsy"
    classes: list[int]
    alpha: float | None = None
    fs: int = 20
    window_s: float = 15.0


class LOSOResult(BaseModel):
    """Pooled out-of-fold predictions from one model."""
    model_name: str
    y_true: list[int]
    y_pred: list[int]
    y_prob: list[float]
    groups: list[str]

    class Config:
        arbitrary_types_allowed = True


# Module-level singleton — import this wherever needed.
CONFIG = AppConfig()

from __future__ import annotations
from dataclasses import dataclass
import carla


@dataclass(frozen=True, slots=True)
class LaneMetrics:
    """Per-tick lane-relative measurements."""
    cte: float              # signed lateral offset from lane centre (m)
    heading_err_deg: float  # heading error vs lane direction (deg)
    lane_width: float       # current lane width (m)
    curvature: float        # road curvature ahead (deg/m)
    on_junction: int        # 1 if on an intersection


_NAN_METRICS = LaneMetrics(
    cte=float("nan"), heading_err_deg=float("nan"),
    lane_width=float("nan"), curvature=float("nan"), on_junction=0,
)


def _normalize_deg(angle: float) -> float:
    """Wrap an angle in degrees to [-180, 180]."""
    while angle > 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle


def compute_lane_metrics(
    world_map: carla.Map,
    vehicle: carla.Vehicle,
) -> LaneMetrics:
    """Compute lane-relative metrics for the current vehicle position."""
    tf: carla.Transform = vehicle.get_transform()
    loc: carla.Location = tf.location
    wp: carla.Waypoint | None = world_map.get_waypoint(
        loc, project_to_road=True, lane_type=carla.LaneType.Driving,
    )
    if wp is None:
        return _NAN_METRICS

    wp_loc: carla.Location = wp.transform.location
    right: carla.Vector3D = wp.transform.get_right_vector()

    dx: float = loc.x - wp_loc.x
    dy: float = loc.y - wp_loc.y
    cte: float = dx * right.x + dy * right.y

    heading_err: float = _normalize_deg(
        tf.rotation.yaw - wp.transform.rotation.yaw
    )

    curvature: float = 0.0
    nxt = wp.next(3.0)
    if nxt:
        dyaw: float = _normalize_deg(
            nxt[0].transform.rotation.yaw - wp.transform.rotation.yaw
        )
        curvature = dyaw / 3.0

    return LaneMetrics(
        cte=cte,
        heading_err_deg=heading_err,
        lane_width=wp.lane_width,
        curvature=curvature,
        on_junction=int(wp.is_junction),
    )

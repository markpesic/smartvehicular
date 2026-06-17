from __future__ import annotations
import argparse
from ..modes.logger_mode import LoggerMode
from ..client_driver import ClientDriver


def main() -> None:
    ap: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", choices=["alert", "drowsy"], required=True)
    ap.add_argument("--sleep", type=float, required=True,
                    help="hours of sleep the prior night")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--town", default="Town04")
    ap.add_argument("--traffic", type=int, default=0)
    ap.add_argument("--low", action="store_true")
    ap.add_argument("--res", default="1280x720")
    args: argparse.Namespace = ap.parse_args()

    resolution: tuple[int, int] = (800, 450) if args.low else tuple(
        int(v) for v in args.res.lower().split("x"))

    mode: LoggerMode = LoggerMode(
        session_type=args.session,
        sleep_hours=args.sleep,
    )
    driver: ClientDriver = ClientDriver(
        mode,
        carla_host=args.host,
        carla_port=args.port,
        town=args.town,
        traffic=args.traffic,
        resolution=resolution,
    )
    driver.run()


if __name__ == "__main__":
    main()

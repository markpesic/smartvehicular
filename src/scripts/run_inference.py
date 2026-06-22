from __future__ import annotations
import argparse
from ..modes.inference_mode import InferenceMode
from ..client_driver import ClientDriver
from ..input_handler import InputType

def main() -> None:
    ap: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--town", default="Town04")
    ap.add_argument("--traffic", type=int, default=0)
    ap.add_argument("--low", action="store_true")
    ap.add_argument("--res", default="1280x720")
    ap.add_argument("--model", default="drowsy_model.pkl")
    ap.add_argument("--meta", default="drowsy_model_meta.json")
    ap.add_argument("--mqtt-host", default="127.0.0.1")
    ap.add_argument("--mqtt-port", type=int, default=1884)
    ap.add_argument("--input", choices=["mouse", "wheel"], default="mouse", help="input device: mouse (default) or wheel")
    ap.add_argument("--debug-axes", action="store_true", help="print live joystick axis values (for finding axis indices)")
    args: argparse.Namespace = ap.parse_args()

    resolution: tuple[int, int] = (800, 450) if args.low else tuple(
        int(v) for v in args.res.lower().split("x"))

    mode: InferenceMode = InferenceMode(
        model_path=args.model,
        meta_path=args.meta,
        mqtt_host=args.mqtt_host,
        mqtt_port=args.mqtt_port,
    )
    driver: ClientDriver = ClientDriver(
        mode,
        carla_host=args.host,
        carla_port=args.port,
        town=args.town,
        traffic=args.traffic,
        resolution=resolution,
        input_type=InputType(args.input),
        debug_axes=args.debug_axes,
    )
    driver.run()


if __name__ == "__main__":
    main()

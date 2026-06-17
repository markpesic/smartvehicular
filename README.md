# Setup & Deployment

## Prerequisites

- [CARLA simulator](https://carla.org/) (0.9.x)
- [Docker](https://www.docker.com/)
- Python 3.10+
- [uv](https://github.com/astral-sh/uv)

## Install

```bash
uv venv
uv pip install -e .
uv pip install path/to/carla-0.9.xx-cpXX-win_amd64.whl
```

## MQTT broker

```bash
cd docker
docker compose up -d
```

Subscribe to messages:

```bash
docker exec -it drowsy-mosquitto mosquitto_sub -t "carla/drowsiness/#" -v
```

Stop:

```bash
docker compose down
```

## Run

Start CARLA first:

```bash
CarlaUE4.exe -quality-level=Low
```

### Data collection

```bash
uv run python -m src.scripts.run_logger --session alert --sleep 8 --town Town04 --low
uv run python -m src.scripts.run_logger --session drowsy --sleep 4.5 --town Town04 --low
```

### Feature extraction

```bash
uv run python -m src.scripts.extract_features --data ./logs --out features.csv
```

### Training

```bash
uv run python -m src.scripts.train_models --features features.csv
```

### Live detection

```bash
uv run python -m src.scripts.run_inference --town Town04 --low
```

## Controls

| Key | Action |
|-----|--------|
| Mouse | Steer |
| W / S / Space | Throttle / brake |
| 1–9 | KSS rating |
| M | Event marker |
| R | Reverse |
| G | Grab/release mouse |
| Tab | Respawn |
| Esc | Quit |

"""Send a batch of random samples from the penguins dataset to the local model server."""

import argparse
import json
import sys
from pathlib import Path

import httpx
import pandas as pd

FEATURES = ["island", "culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g", "sex"]
DEFAULT_URL = "http://0.0.0.0:8080/invocations"
DEFAULT_N = 5


def load_samples(csv_path: Path, n: int) -> list[dict]:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=FEATURES)
    return df[FEATURES].sample(n=n).to_dict(orient="records")


def send(samples: list[dict], url: str) -> dict:
    response = httpx.post(
        url,
        headers={"Content-Type": "application/json"},
        content=json.dumps({"inputs": samples}),
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="Send sample requests to the local model server.")
    parser.add_argument("--csv", default="data/penguins.csv", help="Path to the penguins CSV file.")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="Number of random samples to send.")
    parser.add_argument("--url", default=DEFAULT_URL, help="Model server invocations URL.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    samples = load_samples(csv_path, args.n)
    print(f"Sending {len(samples)} sample(s) to {args.url} ...")
    print(json.dumps({"inputs": samples}, indent=2))

    predictions = send(samples, args.url)
    print("\nPredictions:")
    print(json.dumps(predictions, indent=2))


if __name__ == "__main__":
    main()

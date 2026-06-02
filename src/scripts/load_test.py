"""Load test the local model server and report performance metrics."""

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import httpx
import pandas as pd

FEATURES = [
    "island",
    "culmen_length_mm",
    "culmen_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
    "sex",
]
DEFAULT_URL = "http://0.0.0.0:8080/invocations"


def load_samples(csv_path: Path, n: int) -> list[dict]:
    df = pd.read_csv(csv_path).dropna(subset=FEATURES)
    return df[FEATURES].sample(n=n, replace=True).to_dict(orient="records")


async def send_request(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    url: str,
    payload: dict,
) -> tuple[float, int]:
    """Return (elapsed_seconds, status_code). Status 0 means connection error."""
    async with semaphore:
        start = time.perf_counter()
        try:
            response = await client.post(
                url,
                headers={"Content-Type": "application/json"},
                content=json.dumps(payload),
                timeout=30,
            )
            elapsed = time.perf_counter() - start
            return elapsed, response.status_code
        except Exception:
            elapsed = time.perf_counter() - start
            return elapsed, 0


def report(results: list[tuple[float, int]], total_elapsed: float) -> None:
    latencies = [r[0] for r in results]
    status_codes = [r[1] for r in results]

    successes = sum(1 for s in status_codes if 200 <= s < 300)
    errors = len(results) - successes
    error_rate = errors / len(results) * 100
    throughput = len(results) / total_elapsed

    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)

    def percentile(p: float) -> float:
        idx = max(0, int(n * p / 100) - 1)
        return sorted_latencies[idx] * 1000

    print("\n── Results ─────────────────────────────────")
    print(f"  Requests      : {len(results)}")
    print(f"  Successful    : {successes}")
    print(f"  Errors        : {errors}  ({error_rate:.1f}%)")
    print(f"  Total time    : {total_elapsed:.2f}s")
    print(f"  Throughput    : {throughput:.1f} req/s")
    print("── Latency (ms) ─────────────────────────────")
    print(f"  Min           : {min(latencies) * 1000:.1f}")
    print(f"  Mean          : {statistics.mean(latencies) * 1000:.1f}")
    print(f"  Median (p50)  : {percentile(50):.1f}")
    print(f"  p95           : {percentile(95):.1f}")
    print(f"  p99           : {percentile(99):.1f}")
    print(f"  Max           : {max(latencies) * 1000:.1f}")
    print("─────────────────────────────────────────────\n")

    if errors:
        breakdown = {}
        for s in status_codes:
            if not (200 <= s < 300):
                label = f"HTTP {s}" if s else "Connection error"
                breakdown[label] = breakdown.get(label, 0) + 1
        print("  Error breakdown:")
        for label, count in breakdown.items():
            print(f"    {label}: {count}")
        print()


async def run(url: str, total: int, concurrency: int, samples_per_request: int, csv: Path) -> None:
    samples = load_samples(csv, max(total, 256))
    semaphore = asyncio.Semaphore(concurrency)

    print(f"Target          : {url}")
    print(f"Requests        : {total}")
    print(f"Concurrency     : {concurrency}")
    print(f"Samples/request : {samples_per_request}")
    print("Warming up...", end=" ", flush=True)

    async with httpx.AsyncClient() as client:
        # Single warm-up request so the first real requests don't skew results.
        await client.post(
            url,
            headers={"Content-Type": "application/json"},
            content=json.dumps({"inputs": [samples[0]]}),
            timeout=30,
        )
        print("done\nRunning...", flush=True)

        tasks = [
            send_request(
                client,
                semaphore,
                url,
                {"inputs": samples[i % len(samples) : i % len(samples) + samples_per_request]},
            )
            for i in range(total)
        ]

        wall_start = time.perf_counter()
        results = await asyncio.gather(*tasks)
        wall_elapsed = time.perf_counter() - wall_start

    report(list(results), wall_elapsed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Load test the local model server.")
    parser.add_argument("--url", default=DEFAULT_URL, help="Invocations endpoint URL.")
    parser.add_argument("--requests", type=int, default=100, help="Total number of requests.")
    parser.add_argument("--concurrency", type=int, default=10, help="Max concurrent requests.")
    parser.add_argument(
        "--samples-per-request",
        type=int,
        default=1,
        help="Number of samples per request payload.",
    )
    parser.add_argument("--csv", default="data/penguins.csv", help="Path to the penguins CSV.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    asyncio.run(
        run(
            url=args.url,
            total=args.requests,
            concurrency=args.concurrency,
            samples_per_request=args.samples_per_request,
            csv=csv_path,
        )
    )


if __name__ == "__main__":
    main()

import csv
import json
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "bench-release" / "bench-release"
OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "benchmark_results.csv"

COLUMNS = [
    "benchmark",
    "source_split",
    "model",
    "run_id",
    "prompt_id",
    "query_text",
    "score",
    "cost_usd",
    "prompt_tokens",
    "completion_tokens",
]


def run_details(path):
    parts = path.relative_to(RAW_DIR).parts

    if len(parts) == 4:
        benchmark, source_split, model, _ = parts
    elif len(parts) == 3:
        benchmark, model, _ = parts
        prefix, separator, _ = path.stem.rpartition(f"-{model}-")
        if not separator or "-" not in prefix:
            raise ValueError(f"Could not identify the split in {path.name}")
        source_split = prefix.split("-", 1)[1]
    else:
        raise ValueError(f"Unexpected folder layout: {path}")

    return benchmark, source_split, model, path.stem


def token_count(value, field, issues):
    if value is None:
        return None
    if type(value) is int and value >= 0:
        return value

    issues[field] += 1
    return None


def main():
    files = sorted(RAW_DIR.rglob("*.json"))
    if not files:
        raise SystemExit(f"No JSON files found in {RAW_DIR}")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    temporary_csv = OUTPUT_CSV.with_suffix(".csv.tmp")
    issues = Counter()
    row_count = 0

    with temporary_csv.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(COLUMNS)

        for file_number, path in enumerate(files, start=1):
            benchmark, source_split, model, run_id = run_details(path)
            with path.open(encoding="utf-8-sig") as source:
                records = json.load(source)["records"]

            for record in records:
                score = record.get("score")
                if score is None:
                    issues["missing scores"] += 1

                writer.writerow([
                    benchmark,
                    source_split,
                    model,
                    run_id,
                    record["index"],
                    record["origin_query"],
                    score,
                    record.get("cost"),
                    token_count(record.get("prompt_tokens"), "prompt_tokens", issues),
                    token_count(
                        record.get("completion_tokens"), "completion_tokens", issues
                    ),
                ])
                row_count += 1

            if file_number % 100 == 0:
                print(f"Processed {file_number:,} of {len(files):,} files")

    temporary_csv.replace(OUTPUT_CSV)
    print(f"Exported {row_count:,} results from {len(files):,} files")
    print(f"CSV: {OUTPUT_CSV}")
    for field, count in sorted(issues.items()):
        print(f"{field}: {count:,} written as blank")


if __name__ == "__main__":
    main()
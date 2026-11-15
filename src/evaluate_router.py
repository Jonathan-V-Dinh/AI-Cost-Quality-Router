"""Evaluate a prompt-only routing policy on cached LLM benchmark outcomes."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

CHEAP = "deepseek-v3.1-terminus"
STRONG = "gpt-5"
BUDGETS = (0, 5, 10, 20, 30, 50, 75, 100)


def read_run(path):
    records = json.loads(path.read_bytes())["records"]
    by_id = {str(row["index"]): row for row in records}
    if len(by_id) != len(records):
        raise ValueError(f"Duplicate prompt IDs in {path.name}")
    return by_id


def paired_results(root, benchmark, split, cheap_run, strong_run):
    cheap = read_run(root / benchmark / split / CHEAP / cheap_run)
    strong = read_run(root / benchmark / split / STRONG / strong_run)
    pairs = []
    seen_questions = set()
    for prompt_id in sorted(cheap.keys() & strong.keys()):
        left, right = cheap[prompt_id], strong[prompt_id]
        question = " ".join(left["origin_query"].split())
        if not question or question != " ".join(right["origin_query"].split()):
            continue
        if any(row.get("score") not in (0, 1) or row.get("cost") is None
               for row in (left, right)):
            continue
        key = question.casefold()
        if key in seen_questions:
            continue
        seen_questions.add(key)
        pairs.append((question, left["score"], right["score"],
                      left["cost"], right["cost"]))
    print(f"{benchmark}/{split}: {len(pairs)} unique usable pairs "
          f"from {len(cheap)} cheap and {len(strong)} strong rows")
    return pairs


def evaluate(pairs, predictions):
    _, cheap, strong, cheap_cost, strong_cost = zip(*pairs)
    cheap, strong = np.array(cheap), np.array(strong)
    cost_gap = np.array(strong_cost) - cheap_cost
    orderings = {
        "learned_gain": np.argsort(-predictions, kind="stable"),
        "longest_prompt_first": np.argsort(
            -np.array([len(pair[0].split()) for pair in pairs]), kind="stable"
        ),
    }
    for budget in BUDGETS:
        upgrades = len(pairs) * budget // 100
        policies = orderings.items()
        if budget in (0, 100):
            policies = [("always_deepseek" if budget == 0 else "always_gpt5",
                         orderings["learned_gain"])]
        for policy, ordering in policies:
            selected = ordering[:upgrades]
            gains = strong[selected] - cheap[selected]
            net = int(gains.sum())
            yield {
                "policy": policy, "budget_pct": budget, "questions": len(pairs),
                "upgrades": upgrades, "rescued": int((gains == 1).sum()),
                "harmed": int((gains == -1).sum()), "net_correct": net,
                "accuracy_pct": round(100 * (cheap.sum() + net) / len(pairs), 4),
                "recorded_usd_per_1000": round(
                    1000 * (sum(cheap_cost) + cost_gap[selected].sum()) / len(pairs), 4
                ),
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()

    development = paired_results(
        args.data_root, "mmlupro", "test_3000",
        "mmlupro-test_3000-deepseek-v3.1-terminus-20251024_104950.json",
        "mmlupro-test_3000-gpt-5-20251025_121135.json",
    )
    router = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        Ridge(alpha=1.0),
    )
    router.fit([row[0] for row in development],
               [row[2] - row[1] for row in development])

    evaluation = paired_results(
        args.data_root, "simpleqa", "test",
        "simpleqa-test-deepseek-v3.1-terminus-20251024_110816.json",
        "simpleqa-test-gpt-5-20251025_131712.json",
    )
    seen = {row[0].casefold() for row in development}
    test = [row for row in evaluation if row[0].casefold() not in seen]
    print(f"Excluded {len(evaluation) - len(test)} exact normalized text overlaps")
    if not test:
        raise ValueError("No aligned nonoverlapping SimpleQA prompts")
    rows = list(evaluate(test, router.predict([row[0] for row in test])))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        if row["budget_pct"] in (0, 20, 100):
            print(row)


if __name__ == "__main__":
    main()

import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

from evaluate_router import TfidfVectorizer, make_pipeline, paired_results

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "bench-release" / "bench-release"
REPORT = ROOT / "reports" / "hle_budget_results.csv"
BUDGET_PER_1000 = 10.0  # Example budget in USD, set before scoring HLE.


def prompt_lengths(pairs):
    return np.array([len(row[0].split()) for row in pairs])


def choose_upgrades(priority, estimated_upgrade_cost, remaining_budget):
    selected = np.zeros(len(priority), dtype=bool)

    if remaining_budget < 0:
        return selected

    for index in np.argsort(-priority, kind="stable"):
        cost = estimated_upgrade_cost[index]
        if priority[index] > 0 and cost <= remaining_budget:
            selected[index] = True
            remaining_budget -= cost

    return selected


def main():
    train = paired_results(
        DATA, "mmlupro", "test_3000",
        "mmlupro-test_3000-deepseek-v3.1-terminus-20251024_104950.json",
        "mmlupro-test_3000-gpt-5-20251025_121135.json",
    )
    hle = paired_results(
        DATA, "hle", "test",
        "hle-test-deepseek-v3.1-terminus-20251024_101122.json",
        "hle-test-gpt-5-20251025_114522.json",
    )
    seen = {row[0].casefold() for row in train}
    test = [row for row in hle if row[0].casefold() not in seen]

    router = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        Ridge(alpha=1.0),
    )
    router.fit([row[0] for row in train], [row[2] - row[1] for row in train])
    predicted_gain = router.predict([row[0] for row in test])

    # Estimate each model's cost from prompt length using MMLU-Pro only.
    cost_model = Ridge(alpha=1.0)
    cost_model.fit(
        np.log1p(prompt_lengths(train)).reshape(-1, 1),
        [[row[3], row[4]] for row in train],
    )
    estimated_costs = np.maximum(
        cost_model.predict(np.log1p(prompt_lengths(test)).reshape(-1, 1)),
        0,
    )
    estimated_deepseek, estimated_gpt5 = estimated_costs.T
    upgrade_cost = np.maximum(estimated_gpt5 - estimated_deepseek, 0)
    cost_for_ranking = np.maximum(upgrade_cost, 0.000001)
    total_budget = BUDGET_PER_1000 * len(test) / 1000
    remaining_budget = total_budget - estimated_deepseek.sum()

    policies = {
        "always_deepseek": np.zeros(len(test), dtype=bool),
        "length_per_estimated_dollar": choose_upgrades(
            prompt_lengths(test) / cost_for_ranking,
            upgrade_cost,
            remaining_budget,
        ),
        "learned_gain_per_estimated_dollar": choose_upgrades(
            predicted_gain / cost_for_ranking,
            upgrade_cost,
            remaining_budget,
        ),
        "always_gpt5": np.ones(len(test), dtype=bool),
    }

    outcomes = np.array([row[1:] for row in test])
    deepseek_score, gpt5_score, deepseek_cost, gpt5_cost = outcomes.T
    rows = []

    for policy, selected in policies.items():
        correct = int(np.where(selected, gpt5_score, deepseek_score).sum())
        recorded_total = np.where(selected, gpt5_cost, deepseek_cost).sum()
        estimated_total = np.where(
            selected, estimated_gpt5, estimated_deepseek
        ).sum()

        rows.append({
            "policy": policy,
            "questions": len(test),
            "gpt5_upgrades": int(selected.sum()),
            "correct": correct,
            "accuracy_pct": round(100 * correct / len(test), 4),
            "budget_usd_per_1000": BUDGET_PER_1000,
            "estimated_usd_per_1000": round(1000 * estimated_total / len(test), 4),
            "recorded_usd_per_1000": round(1000 * recorded_total / len(test), 4),
            "met_budget": bool(recorded_total <= total_budget),
        })

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Compared {len(test):,} HLE questions. Results saved to {REPORT}")


if __name__ == "__main__":
    main()
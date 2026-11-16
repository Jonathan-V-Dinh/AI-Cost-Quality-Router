import csv
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

from evaluate_router import paired_results

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "bench-release" / "bench-release"
REPORT = ROOT / "reports" / "gpqa_equal_spend.csv"

BUDGETS = (10, 20, 40)  # USD per 1,000 requests
SEED = 42
RANDOM_TRIALS = 200


def word_counts(rows):
    return np.array([len(row[0].split()) for row in rows])


def affordable_prefix(order, upgrade_costs, remaining_budget):
    """Take upgrades in priority order until the next one exceeds the cap."""
    if remaining_budget < 0:
        return order[:0]

    cumulative_cost = np.cumsum(upgrade_costs[order])
    over_budget = np.flatnonzero(cumulative_cost > remaining_budget)
    stop = over_budget[0] if over_budget.size else len(order)
    return order[:stop]


def main():
    train = paired_results(
        DATA, "mmlupro", "test_3000",
        "mmlupro-test_3000-deepseek-v3.1-terminus-20251024_104950.json",
        "mmlupro-test_3000-gpt-5-20251025_121135.json",
    )
    gpqa = paired_results(
        DATA, "gpqa", "test",
        "gpqa-test-deepseek-v3.1-terminus-20251024_090057.json",
        "gpqa-test-gpt-5-20251019_024120.json",
    )

    seen = {row[0].casefold() for row in train}
    fresh = [row for row in gpqa if row[0].casefold() not in seen]
    if len(fresh) < 10:
        raise ValueError("Too few matched GPQA prompts for a split.")

    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(fresh))
    cutoff = int(0.4 * len(fresh))
    calibration = [fresh[index] for index in indices[:cutoff]]
    test = [fresh[index] for index in indices[cutoff:]]
    print(f"GPQA: {len(calibration)} calibration, {len(test)} test prompts")

    router = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        Ridge(alpha=1.0),
    )
    router.fit([row[0] for row in train], [row[2] - row[1] for row in train])
    predicted_gain = router.predict([row[0] for row in test])

    cost_model = Ridge(alpha=1.0)
    cost_model.fit(
        np.log1p(word_counts(calibration)).reshape(-1, 1),
        [[row[3], row[4]] for row in calibration],
    )
    lengths = word_counts(test)
    estimated_cheap, estimated_strong = np.maximum(
        cost_model.predict(np.log1p(lengths).reshape(-1, 1)), 0
    ).T
    estimated_gap = estimated_strong - estimated_cheap
    ranking_cost = np.maximum(estimated_gap, 0.000001)

    learned_order = np.argsort(-predicted_gain / ranking_cost, kind="stable")
    learned_order = learned_order[predicted_gain[learned_order] > 0]
    orders = {
        "learned_gain_per_dollar": [learned_order],
        "length_per_dollar": [
            np.argsort(-lengths / ranking_cost, kind="stable")
        ],
        "random_upgrades": [rng.permutation(len(test)) for _ in range(RANDOM_TRIALS)],
        "always_deepseek": [np.array([], dtype=int)],
    }

    cheap_score, strong_score, cheap_cost, strong_cost = np.array(
        [row[1:] for row in test]
    ).T
    recorded_gap = strong_cost - cheap_cost
    n = len(test)
    rows = []

    for budget in BUDGETS:
        cap = budget * n / 1000
        budget_modes = {
            "estimated_cost": (estimated_cheap.sum(), estimated_gap),
            "recorded_cost_replay": (cheap_cost.sum(), recorded_gap),
        }
        for mode, (base_cost, gap) in budget_modes.items():
            for policy, rankings in orders.items():
                results = []
                for order in rankings:
                    selected = affordable_prefix(order, gap, cap - base_cost)
                    correct = cheap_score.sum() + (
                        strong_score[selected] - cheap_score[selected]
                    ).sum()
                    spent = cheap_cost.sum() + recorded_gap[selected].sum()
                    results.append((len(selected), correct, spent, spent <= cap))

                upgrades, correct, spent, met_cap = np.mean(results, axis=0)
                rows.append({
                    "budget_mode": mode,
                    "policy": policy,
                    "test_questions": n,
                    "budget_usd_per_1000": budget,
                    "mean_upgrades": round(upgrades, 2),
                    "mean_correct": round(correct, 2),
                    "accuracy_pct": round(100 * correct / n, 2),
                    "recorded_usd_per_1000": round(1000 * spent / n, 4),
                    "trials_within_cap_pct": round(100 * met_cap, 2),
                })

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Always GPT-5 on GPQA test: {100 * strong_score.mean():.2f}% "
          f"at ${1000 * strong_cost.mean():.4f} per 1,000 requests")
    print(f"Saved {REPORT}")


if __name__ == "__main__":
    main()
-- Inventory model runs before comparing performance on shared prompts
SELECT
    benchmark,
    source_split,
    model,
    run_id,
    COUNT(*) AS prompt_count,
    COUNT(score) AS scored_prompts,
    COUNT(cost_usd) AS prompts_with_cost,
    COUNT(*) FILTER (WHERE cost_usd = 0) AS zero_cost_prompts
FROM public.benchmark_results
GROUP BY benchmark, source_split, model, run_id
ORDER BY
    CASE WHEN benchmark = 'mmlupro' THEN 0 ELSE 1 END,
    benchmark,
    source_split,
    model,
    run_id;

-- Cost and accuracy comparison:
-- Business question: If we used one model for every request, how much would it cost per 1,000 requests, and how often would it answer correctly?
-- Purpose: Establish the fixed model baselines our ML router will need to beat.
-- I'm using the MMLU-Pro benchmark for this initial test
-- GPT-5 had the highest accuracy at 87.37% with a cost of $12.06 per 1,000 requests
-- DeepSeek V3.1 Terminus had the lowest cost with $0.56 per 1000 requests, but the accuracy was 84.03%
-- Result saved as mmlu_pro_initial_test.csv
WITH eligible_runs AS (
    SELECT model, run_id
    FROM public.benchmark_results
    WHERE benchmark = 'mmlupro'
      AND source_split = 'test_3000'
    GROUP BY model, run_id
    HAVING COUNT(*) = 3000
       AND COUNT(*) FILTER (WHERE score IN (0, 1)) = 3000
       AND COUNT(cost_usd) = 3000
),
selected_runs AS (
    SELECT DISTINCT ON (model) model, run_id
    FROM eligible_runs
    ORDER BY model, RIGHT(run_id, 15) DESC, run_id DESC
),
results AS (
    SELECT b.model, b.run_id, b.prompt_id, b.query_text, b.score, b.cost_usd
    FROM public.benchmark_results AS b
    JOIN selected_runs AS r USING (model, run_id)
    WHERE b.benchmark = 'mmlupro'
      AND b.source_split = 'test_3000'
),
shared_prompts AS (
    SELECT prompt_id
    FROM results
    GROUP BY prompt_id
    HAVING COUNT(*) = (SELECT COUNT(*) FROM selected_runs)
       AND COUNT(DISTINCT query_text) = 1
)
SELECT
    r.model,
    CASE WHEN r.model = 'openrouter' THEN 'router' ELSE 'model' END AS entry_type,
    r.run_id,
    COUNT(*) AS shared_questions,
    ROUND(100 * AVG(r.score)::numeric, 2) AS accuracy_pct,
    ROUND(1000 * AVG(r.cost_usd), 4) AS recorded_usd_per_1000,
    COUNT(*) FILTER (WHERE r.cost_usd = 0) AS zero_cost_rows
FROM results AS r
JOIN shared_prompts USING (prompt_id)
GROUP BY r.model, r.run_id
ORDER BY accuracy_pct DESC, recorded_usd_per_1000 ASC;

-- Prompt level comparison for the first routing candidate pair (GPT 5 vs DeepSeek Terminus)
-- Count how often only GPT-5 or only DeepSeek answers a shared question correctly.
 WITH results AS MATERIALIZED (
    SELECT model, prompt_id, query_text, score
    FROM public.benchmark_results
    WHERE benchmark = 'mmlupro'
      AND source_split = 'test_3000'
      AND run_id IN (
          'mmlupro-test_3000-gpt-5-20251025_121135',
          'mmlupro-test_3000-deepseek-v3.1-terminus-20251024_104950'
      )
)
SELECT
    COUNT(*) AS shared_questions,
    COUNT(*) FILTER (
        WHERE cheap.score = 1 AND strong.score = 1
    ) AS both_correct,
    COUNT(*) FILTER (
        WHERE cheap.score = 0 AND strong.score = 1
    ) AS only_gpt5_correct,
    COUNT(*) FILTER (
        WHERE cheap.score = 1 AND strong.score = 0
    ) AS only_deepseek_correct,
    COUNT(*) FILTER (
        WHERE cheap.score = 0 AND strong.score = 0
    ) AS both_wrong,
    ROUND(
        100.0 * COUNT(*) FILTER (WHERE cheap.score = 1 OR strong.score = 1)
        / NULLIF(COUNT(*), 0), 2
    ) AS accuracy_ceiling_pct
FROM results AS cheap
JOIN results AS strong USING (prompt_id)
WHERE cheap.model = 'deepseek-v3.1-terminus'
  AND strong.model = 'gpt-5'
  AND cheap.query_text COLLATE "C" = strong.query_text COLLATE "C";
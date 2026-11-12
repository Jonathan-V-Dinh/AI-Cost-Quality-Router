\copy public.benchmark_results (benchmark, source_split, model, run_id, prompt_id, query_text, score, cost_usd, prompt_tokens, completion_tokens) FROM 'data/processed/benchmark_results.csv' WITH (FORMAT csv, HEADER true, ENCODING 'UTF8')

SELECT COUNT(*) AS loaded_rows,
       COUNT(DISTINCT benchmark) AS benchmarks,
       COUNT(DISTINCT model) AS models,
       COUNT(*) FILTER (WHERE score IS NULL) AS missing_scores
FROM public.benchmark_results;
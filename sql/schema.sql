-- One row per prompt, model, and benchmark run
CREATE TABLE IF NOT EXISTS public.benchmark_results (
    benchmark         text NOT NULL,
    source_split      text NOT NULL,
    model             text NOT NULL,
    run_id            text NOT NULL,
    prompt_id         text NOT NULL,
    query_text        text NOT NULL,
    score             double precision,
    cost_usd          numeric CHECK (cost_usd >= 0),
    prompt_tokens     integer CHECK (prompt_tokens >= 0),
    completion_tokens integer CHECK (completion_tokens >= 0),
    PRIMARY KEY (benchmark, source_split, model, run_id, prompt_id)
);
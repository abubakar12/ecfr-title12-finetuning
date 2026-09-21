# Results: eCFR Title 12 LoRA Post-Training

- Base model: `meta-llama/Meta-Llama-3.1-8B-Instruct`
- Held-out eval examples: 300 (section-level split, greedy decoding)
- Evaluated: 2026-09-20T20:20:05Z
- Device: cuda (44.4 GB), dtype bfloat16

## Overall (deltas vs untuned base; ⚠ = regression)

| checkpoint | n | citation_accuracy | wrong_citation_rate | token_f1 | rouge_l |
|---|---|---|---|---|---|
| base | 300 | 0.8000 | 0.1833 | 0.2822 | 0.2014 |
| cpt | 300 | 0.7867 (-0.0133 ⚠) | 0.2000 (+0.0167 ⚠) | 0.3236 (+0.0414) | 0.2757 (+0.0743) |
| sft | 300 | 0.8000 (+0.0000) | 0.2000 (+0.0167 ⚠) | 0.5213 (+0.2391) | 0.4524 (+0.2510) |
| dpo | 300 | 0.8000 (+0.0000) | 0.1967 (+0.0134 ⚠) | 0.4986 (+0.2164) | 0.4243 (+0.2229) |
| grpo | 300 | 0.8000 (+0.0000) | 0.2000 (+0.0167 ⚠) | 0.5249 (+0.2427) | 0.4558 (+0.2544) |

## token_f1 by question type

| type | base | cpt | sft | dpo | grpo |
|---|---|---|---|---|---|
| citation_lookup | 0.4099 | 0.3477 | 0.6315 | 0.5329 | 0.6304 |
| definition | 0.2804 | 0.3485 | 0.5402 | 0.5042 | 0.5401 |
| overview | 0.2513 | 0.3609 | 0.5400 | 0.5207 | 0.5480 |
| provisions | 0.2417 | 0.2705 | 0.4400 | 0.4584 | 0.4431 |

## Training stages

- **SFT** — model `meta-llama/Meta-Llama-3.1-8B-Instruct`, n_train=14722, trained 2026-09-20T16:09:39Z
- **DPO** — model `meta-llama/Meta-Llama-3.1-8B-Instruct`, n_pairs=10374, trained 2026-09-20T17:54:56Z
- **GRPO** — model `meta-llama/Meta-Llama-3.1-8B-Instruct`, n_prompts=10374, trained 2026-09-20T19:39:06Z

Artifacts: `results/eval_results.json`, `results/report.md`, `results/model_report.pdf`, per-checkpoint generations in `results/generations_*.jsonl`.

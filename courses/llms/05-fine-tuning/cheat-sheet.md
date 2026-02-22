# Fine-Tuning Cheat Sheet

## Decision Tree

```
Can you solve the problem with better prompting?
│
├── YES → Prompt engineering. Stop here.
│
└── NO → Does the model need external/private knowledge?
    │
    ├── YES → RAG (retrieval-augmented generation)
    │         Consider combining with fine-tuning for output format/style.
    │
    └── NO → Do you need to change model behavior (style, format, reasoning)?
        │
        ├── YES → Fine-tuning
        │   │
        │   ├── Small behavior change (format, tone) → LoRA, 50-500 examples
        │   ├── Domain adaptation → LoRA/QLoRA, 500-5K examples
        │   ├── Complex task learning → LoRA, 1K-10K examples
        │   └── Deep domain knowledge → Continued pre-training + LoRA
        │
        └── NO → Do you need a smaller/cheaper/faster model?
            │
            ├── YES → Distillation (train small model on large model outputs)
            │
            └── NO → Re-examine your prompt engineering approach.
```

---

## Fine-Tuning Method Comparison

| Method | Params Trained | GPU RAM (7B) | GPU RAM (70B) | Min Data | Quality Ceiling | Training Speed |
|---|---|---|---|---|---|---|
| **Full fine-tune** | 100% | ~56GB | ~560GB | 5K+ | Highest | Slowest |
| **LoRA (r=16)** | ~0.5% | ~16GB | ~160GB | 200+ | Very high | Fast |
| **QLoRA (r=16)** | ~0.5% | ~8GB | ~40GB | 200+ | High | Fast |
| **Prefix tuning** | ~0.1% | ~15GB | ~150GB | 500+ | Moderate | Very fast |
| **IA3** | ~0.01% | ~14GB | ~140GB | 500+ | Moderate | Fastest |

---

## Data Format Templates

### OpenAI Chat Format (JSONL)

```json
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Summarize this text: ..."}, {"role": "assistant", "content": "Here is the summary: ..."}]}
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Translate to French: ..."}, {"role": "assistant", "content": "Voici la traduction: ..."}]}
```

### Alpaca Format (JSON)

```json
[
  {
    "instruction": "Classify the sentiment of this review.",
    "input": "The product arrived damaged and customer service was unhelpful.",
    "output": "Negative"
  },
  {
    "instruction": "Summarize the following text in one sentence.",
    "input": "Long text here...",
    "output": "Summary here."
  }
]
```

### ShareGPT Format (Multi-Turn)

```json
[
  {
    "conversations": [
      {"from": "human", "value": "What is LoRA?"},
      {"from": "gpt", "value": "LoRA stands for Low-Rank Adaptation..."},
      {"from": "human", "value": "When should I use it?"},
      {"from": "gpt", "value": "Use LoRA when you want to fine-tune..."}
    ]
  }
]
```

### DPO Preference Pairs (JSONL)

```json
{"prompt": "Explain machine learning simply.", "chosen": "Machine learning is when computers learn patterns from data...", "rejected": "ML is a subset of AI utilizing statistical methodologies..."}
```

---

## GPU Requirements Quick Reference

### LoRA Fine-Tuning

| Model Size | Min GPU | Recommended GPU | Batch Size 8 |
|---|---|---|---|
| 1.5B | RTX 3080 (10GB) | RTX 4090 (24GB) | ~10GB |
| 7B | RTX 4090 (24GB) | A100-40GB | ~18GB |
| 13B | A100-40GB | A100-80GB | ~32GB |
| 70B | 2x A100-80GB | 4x A100-80GB | ~170GB |

### QLoRA Fine-Tuning

| Model Size | Min GPU | Recommended GPU | Batch Size 8 |
|---|---|---|---|
| 1.5B | RTX 3060 (8GB) | RTX 3080 (10GB) | ~5GB |
| 7B | RTX 3090 (24GB) | RTX 4090 (24GB) | ~10GB |
| 13B | RTX 4090 (24GB) | A100-40GB | ~16GB |
| 70B | A100-80GB | A100-80GB | ~48GB |

---

## Training Hyperparameter Starting Points

| Parameter | LoRA | QLoRA | Full Fine-Tune |
|---|---|---|---|
| Learning rate | 2e-4 | 2e-4 | 2e-5 |
| Epochs | 3 | 3 | 1-3 |
| Batch size (effective) | 32 | 32 | 32-128 |
| Micro batch size | 4-8 | 4-8 | 4-16 |
| Gradient accumulation | 4-8 | 4-8 | 2-8 |
| Warmup ratio | 0.03 | 0.03 | 0.03 |
| Weight decay | 0.01 | 0.01 | 0.01 |
| LR scheduler | cosine | cosine | cosine |
| Max sequence length | Match your data | Match your data | Match your data |

### LoRA-Specific Parameters

| Parameter | Default | Notes |
|---|---|---|
| Rank (r) | 16 | 8 for simple, 32-64 for complex tasks |
| Alpha | 2x rank (32) | Scaling factor; alpha/rank = effective scaling |
| Target modules | q,k,v,o projections | Add gate,up,down projections for more capacity |
| Dropout | 0.05 | 0.1 for small datasets, 0 for large |

---

## Cost Estimates

### API Fine-Tuning

| Provider | Model | Cost per 1M Training Tokens | Min Cost (1K examples) |
|---|---|---|---|
| OpenAI | GPT-4o-mini | ~$3 | ~$2-3 |
| OpenAI | GPT-4o | ~$25 | ~$15-25 |
| Together AI | Llama 3.1 8B | ~$2 | ~$1-2 |

### Cloud GPU Training

| Scenario | GPU | Hours | Cost |
|---|---|---|---|
| 7B LoRA, 1K examples | 1x A100-80GB | 0.5h | $1-3 |
| 7B LoRA, 10K examples | 1x A100-80GB | 3-5h | $10-25 |
| 7B QLoRA, 10K examples | 1x A6000 | 4-8h | $8-16 |
| 13B LoRA, 10K examples | 1x A100-80GB | 6-10h | $20-50 |
| 70B QLoRA, 10K examples | 1x A100-80GB | 24-48h | $80-200 |
| 70B full fine-tune, 50K examples | 8x A100-80GB | 24-72h | $5K-15K |

### Synthetic Data Generation

| Examples | Teacher Model | Cost |
|---|---|---|
| 1,000 | GPT-4o | $5-20 |
| 1,000 | GPT-4o-mini | $1-3 |
| 10,000 | GPT-4o-mini | $5-20 |
| 50,000 | GPT-4o-mini | $25-100 |

---

## Evaluation Metrics Quick Reference

| Metric | Formula / Description | Range | Use For |
|---|---|---|---|
| **Exact Match** | predicted == expected | 0-1 | Classification, extraction |
| **F1** | 2 * (P * R) / (P + R) | 0-1 | Classification |
| **BLEU** | N-gram precision vs reference | 0-1 | Translation, short generation |
| **ROUGE-L** | Longest common subsequence overlap | 0-1 | Summarization |
| **Perplexity** | exp(average cross-entropy loss) | 1-inf (lower=better) | General language quality |
| **Pass@k** | P(at least 1 correct in k samples) | 0-1 | Code generation |
| **LLM-as-Judge** | Strong model rates output (1-5 scale) | 1-5 | Style, helpfulness, quality |

---

## Common Failure Modes

| Problem | Symptoms | Fix |
|---|---|---|
| **Overfitting** | Train loss low, val loss high | Fewer epochs, more data, lower rank, more dropout |
| **Underfitting** | Both losses high | Higher rank, more epochs, higher learning rate |
| **Catastrophic forgetting** | Task-specific quality up, general quality down | Use LoRA, add replay data, lower learning rate |
| **Mode collapse** | Model gives same output for different inputs | More diverse data, lower learning rate, fewer epochs |
| **Format regression** | Model stops following instruction format | Include format examples in training data |
| **Divergence** | Loss spikes or goes to NaN | Lower learning rate, check data for corruption |

---

## Checklist: Before You Fine-Tune

```
[ ] Confirmed that prompt engineering alone doesn't meet quality bar
[ ] Confirmed that RAG doesn't solve the problem (if knowledge-related)
[ ] Collected and validated at least 200 high-quality examples
[ ] Established evaluation metrics and baseline (base model + best prompt)
[ ] Prepared train/validation/test split (80/10/10)
[ ] Checked data for PII, bias, and harmful content
[ ] Chose method (LoRA/QLoRA/full) based on compute budget
[ ] Selected model size based on quality needs and inference budget
[ ] Set up evaluation pipeline (automated + spot-check)
```

## Checklist: After Fine-Tuning

```
[ ] Compared fine-tuned model vs baseline on test set
[ ] Ran general capability benchmarks (check for forgetting)
[ ] Ran safety evaluation (refusal tests, bias checks)
[ ] Tested with production-representative inputs (not just test set)
[ ] Quantized model if deploying on limited hardware
[ ] Evaluated quantized model (don't assume quality is preserved)
[ ] Documented training config, data, and results for reproducibility
[ ] Set up monitoring for production quality
```

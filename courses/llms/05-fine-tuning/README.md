# Fine-Tuning & Model Customization

## The 30-Second Explanation

Fine-tuning takes a pre-trained LLM and further trains it on your specific data to improve performance on your specific task. It's one of several strategies for customizing model behavior -- and often not the first one you should reach for. The decision of *when* to fine-tune (versus prompt engineering, RAG, or long context) is the most important thing to get right.

---

## Decision Framework: How to Customize Model Behavior

Before spending weeks on a fine-tuning pipeline, walk through this decision tree:

```
Is the model already good at the task with the right prompt?
├── YES → Use prompt engineering. You're done.
└── NO → Does the model need access to specific/private knowledge?
    ├── YES → Does the knowledge change frequently?
    │   ├── YES → Use RAG. Fine-tuning bakes in stale knowledge.
    │   └── NO → Does it fit in the context window?
    │       ├── YES → Try long-context prompting first. Cheaper, faster iteration.
    │       └── NO → Use RAG, possibly combined with fine-tuning for style/format.
    └── NO → Do you need to change HOW the model responds (style, format, tone, reasoning pattern)?
        ├── YES → Fine-tuning is likely the right choice.
        └── NO → Are you optimizing for latency or cost at scale?
            ├── YES → Fine-tune a smaller model to match a larger one (distillation).
            └── NO → Revisit prompt engineering with more effort.
```

### Approach Comparison

| Approach | When to Use | Data Needed | Cost | Latency Impact | Maintenance |
|---|---|---|---|---|---|
| **Prompt engineering** | Model can do it, just needs better instructions | 0 (just examples in prompt) | Zero upfront | Slightly higher (longer prompts) | Low -- update prompts |
| **Long context** | Need to include specific documents/data | Documents that fit in context | Per-request token cost | Higher (processing long input) | Low |
| **RAG** | Need access to large/changing knowledge bases | Corpus + embedding pipeline | Moderate (infra + retrieval) | +200-500ms retrieval | Medium (keep index fresh) |
| **Fine-tuning** | Need to change model behavior/style/format | 100-100K+ examples | High upfront ($50-$50K+) | Lower (shorter prompts needed) | High (retrain on new base models) |
| **Continued pre-training** | Need deep domain knowledge (medical, legal) | Millions of domain tokens | Very high ($10K-$500K+) | Neutral | Very high |

### The Real-World Decision

In practice, most production systems layer multiple approaches:

1. **Start with prompt engineering.** Iterate on prompts with 10-20 examples. If this gets you to 85%+ quality, stop.
2. **Add RAG** if the model needs external knowledge. This handles the "what does the model know" problem.
3. **Fine-tune** if you need to change "how the model behaves" -- output format, tone, reasoning style, domain-specific patterns. Fine-tuning handles the "how does the model respond" problem.
4. **Distill** if you need a smaller/faster/cheaper model that preserves the behavior of a larger one.

**Interview insight:** The best answer to "when would you fine-tune?" is not "always" or "never." It's demonstrating you understand the trade-offs and would exhaust cheaper approaches first.

---

## Fine-Tuning Approaches

### Full Fine-Tuning

Update all model weights during training.

**How it works:**
- Load the entire model into GPU memory
- Train on your dataset, updating every parameter
- Save the complete model (same size as original)

**Requirements:**
- A 7B model needs ~56GB VRAM in FP16 (parameters + gradients + optimizer states)
- A 70B model needs ~560GB VRAM -- that's 8x A100-80GB GPUs minimum
- Lots of high-quality training data (10K+ examples typically)

**When to use:**
- You have massive compute budget and large datasets
- Maximum quality is required and you can afford the infrastructure
- You're building a foundation model or doing continued pre-training

**Risks:**
- **Catastrophic forgetting** -- the model loses general capabilities while learning your specific task
- Expensive to iterate -- each experiment takes hours/days
- You now own and maintain a full model

**In practice:** Full fine-tuning is rarely the right choice for applied engineers. LoRA gets you 90-95% of the quality at a fraction of the cost.

---

### LoRA (Low-Rank Adaptation)

The dominant fine-tuning method for applied work. Freeze the base model, train small adapter matrices.

**Core idea:**
Instead of updating the full weight matrix W (which might be 4096x4096), you decompose the update into two small matrices:

```
W_new = W_original + A * B

Where:
  W_original: frozen (4096 x 4096 = 16.7M params)
  A: trainable (4096 x r)
  B: trainable (r x 4096)

If r = 16:
  Trainable params = 4096*16 + 16*4096 = 131K (instead of 16.7M)
  That's 0.8% of the original parameters
```

**Why low rank works:**
Weight updates during fine-tuning tend to be low-rank -- they capture a small number of task-specific patterns, not a complete restructuring of the model's knowledge. Rank-16 to rank-64 captures the vast majority of meaningful updates for most tasks.

**Choosing the rank:**

| Rank | Trainable Params (7B model) | Quality | Use Case |
|---|---|---|---|
| 8 | ~17M (0.24%) | Good for simple tasks | Classification, simple format changes |
| 16 | ~34M (0.49%) | Strong default | Most fine-tuning tasks |
| 32 | ~67M (0.96%) | Higher capacity | Complex tasks, multi-domain |
| 64 | ~134M (1.93%) | Near full fine-tune | Tasks requiring significant behavior change |
| 128+ | ~268M+ (3.8%+) | Diminishing returns | Rarely needed |

**Which layers to target:**
- `q_proj`, `v_proj` -- minimum for reasonable results
- `q_proj`, `k_proj`, `v_proj`, `o_proj` -- standard recommendation
- All attention + MLP layers -- maximum coverage, more params to train

**GPU requirements (LoRA on a 7B model):**
- FP16 base + LoRA adapters: ~16GB VRAM (single A100-40GB or RTX 4090)
- Training throughput: ~100-500 examples/minute depending on sequence length

**Advantages:**
- 10-100x less compute than full fine-tuning
- Multiple LoRA adapters can share one base model (swap adapters at inference)
- Less prone to catastrophic forgetting (base weights are frozen)
- Fast to iterate -- minutes to hours, not hours to days

---

### QLoRA (Quantized LoRA)

LoRA, but the base model is loaded in 4-bit precision. This slashes memory requirements.

**How it works:**
1. Load base model in 4-bit quantization (NF4 format)
2. Add LoRA adapters in FP16/BF16
3. Train only the LoRA adapters
4. Gradients flow through the quantized base model

**GPU requirements (QLoRA on a 7B model):**
- ~6-8GB VRAM -- fits on a consumer RTX 3090/4090
- A 70B model fits on a single A100-80GB with QLoRA

**Quality trade-off:**
- QLoRA typically achieves 95-99% of LoRA quality
- The quality gap narrows with more training data
- Some tasks show no measurable difference

**When to use QLoRA over LoRA:**
- You're GPU-constrained (consumer hardware or limited cloud budget)
- You're experimenting and want fast iteration
- The model is large relative to your available VRAM

**When to prefer LoRA over QLoRA:**
- Maximum quality matters and you have the VRAM
- Training speed matters (quantized ops can be slower per step)
- You're doing a final production training run

---

### Other Adapter Methods

**Prefix Tuning:**
- Prepend trainable "virtual tokens" to the input at each layer
- Very parameter-efficient (~0.1% of params)
- Works well for generation tasks
- Eats into your effective context window

**Prompt Tuning:**
- Similar to prefix tuning but only adds tokens to the input embedding layer
- Even fewer parameters
- Works for simpler tasks, degrades on complex ones

**IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations):**
- Learns scaling vectors (not matrices) for keys, values, and FFN layers
- 10x fewer parameters than LoRA
- Faster training, but lower quality ceiling
- Good for quick experiments

**Comparison:**

| Method | Params Trained | Quality Ceiling | Training Speed | Best For |
|---|---|---|---|---|
| Full fine-tune | 100% | Highest | Slowest | Maximum quality, large budgets |
| LoRA | 0.5-2% | Very high | Fast | Default choice for most tasks |
| QLoRA | 0.5-2% | High | Fast | GPU-constrained environments |
| Prefix tuning | ~0.1% | Moderate | Very fast | Generation, limited compute |
| IA3 | ~0.01% | Moderate | Fastest | Quick experiments, simple tasks |

---

## Data Preparation

Data quality is the single most important factor in fine-tuning success. A model trained on 500 high-quality examples will outperform one trained on 50,000 noisy examples.

### Data Formats

**OpenAI Chat Format (most common for API fine-tuning):**
```json
{"messages": [
  {"role": "system", "content": "You are a helpful legal assistant."},
  {"role": "user", "content": "What is the statute of limitations for breach of contract in California?"},
  {"role": "assistant", "content": "In California, the statute of limitations for breach of a written contract is 4 years (CCP 337). For oral contracts, it is 2 years (CCP 339)."}
]}
```

**Alpaca Format (common for open-source fine-tuning):**
```json
{
  "instruction": "Summarize the following legal document in plain English.",
  "input": "WHEREAS, the Party of the First Part...",
  "output": "This agreement states that..."
}
```

**ShareGPT Format (multi-turn conversations):**
```json
{
  "conversations": [
    {"from": "human", "value": "What is LoRA?"},
    {"from": "gpt", "value": "LoRA (Low-Rank Adaptation) is..."},
    {"from": "human", "value": "How does it compare to full fine-tuning?"},
    {"from": "gpt", "value": "Compared to full fine-tuning..."}
  ]
}
```

### Data Quality Checklist

1. **Correctness** -- Every response should be factually accurate. One wrong example can teach persistent bad habits.
2. **Consistency** -- Same types of inputs should produce same style of outputs. Mixed conventions confuse the model.
3. **Diversity** -- Cover the full range of inputs the model will see in production. Edge cases matter.
4. **Completeness** -- Responses should be complete, not truncated. Include the full expected output.
5. **Deduplication** -- Exact and near-duplicates waste training compute and overfit on repeated patterns.
6. **Length distribution** -- Match the output length distribution you want in production. If your training examples are all 500 tokens, the model will learn to produce ~500 token responses.

### How Much Data Do You Need?

| Task Complexity | Examples Needed | Notes |
|---|---|---|
| Style/format change | 50-200 | "Respond in bullet points" or "Use formal tone" |
| Classification | 100-500 per class | More classes = more data |
| Domain adaptation | 500-5,000 | Medical, legal, technical writing |
| Complex reasoning | 1,000-10,000 | Multi-step tasks, nuanced judgment |
| General-purpose improvement | 10,000-100,000 | Competing with the base model's breadth |

**Rule of thumb:** Start with 200-500 high-quality examples. Measure. Add more only if quality metrics plateau.

### Synthetic Data Generation

Using a stronger model (GPT-4, Claude) to generate training data for a smaller model:

**Why it works:**
- Stronger models produce high-quality examples at scale
- You can generate thousands of examples in hours
- Cost of API calls is typically much less than human annotation

**Key principles:**
- **Diversity is critical** -- vary your generation prompts to avoid homogeneous data
- **Filter aggressively** -- not every generated example is good; score and filter
- **Human review a sample** -- spot-check 5-10% to calibrate quality
- **Avoid model collapse** -- don't train on data that's too uniform or self-referential

**Cost estimate for synthetic data:**
- 1,000 examples at ~500 tokens each = ~500K tokens
- GPT-4 pricing: ~$5-15 for input+output
- Claude pricing: similar range
- This is trivially cheap compared to human annotation

---

## Training Fundamentals

You don't need to be an ML researcher, but you need to understand these concepts to make informed decisions and debug training runs.

### Key Hyperparameters

| Parameter | What It Controls | Typical Range | Starting Point |
|---|---|---|---|
| **Learning rate** | How much weights change per step | 1e-5 to 5e-4 | 2e-4 for LoRA, 2e-5 for full fine-tune |
| **Epochs** | Full passes through the dataset | 1-5 | 3 for small datasets, 1 for large |
| **Batch size** | Examples processed together | 4-128 | 8-32 (larger = smoother but more VRAM) |
| **Warmup ratio** | Fraction of steps with increasing LR | 0.03-0.1 | 0.03 |
| **Weight decay** | Regularization strength | 0-0.1 | 0.01 |
| **Max sequence length** | Longest input+output in tokens | 512-8192 | Match your data distribution |

### Learning Rate

The most important hyperparameter. Too high and the model diverges (loss spikes). Too low and it learns nothing (loss barely moves).

```
Learning rate too high:   loss jumps around, never converges
Learning rate too low:    loss decreases extremely slowly, might not converge
Learning rate just right: loss decreases steadily, then plateaus
```

**LoRA learning rates are higher than full fine-tuning** because you're updating fewer parameters. A typical LoRA learning rate is 1e-4 to 5e-4, while full fine-tuning uses 1e-5 to 5e-5.

### Overfitting Detection

```
Training loss ↓ but Validation loss ↑  =  OVERFITTING

Step  | Train Loss | Val Loss  | Status
------|------------|-----------|--------
100   | 2.1        | 2.2       | Normal -- both decreasing
500   | 1.4        | 1.5       | Normal -- both decreasing
1000  | 0.8        | 1.3       | Warning -- gap widening
2000  | 0.3        | 1.8       | Overfitting -- stop training
```

**Mitigation:**
- Reduce epochs (you've trained too long)
- Increase dataset size (more data = harder to memorize)
- Reduce LoRA rank (fewer trainable parameters = less capacity to memorize)
- Add regularization (weight decay, dropout)

### Train/Validation/Test Split

```
Total dataset
├── Training set (80%)     -- model learns from this
├── Validation set (10%)   -- monitor during training, tune hyperparameters
└── Test set (10%)         -- evaluate final model, touch only once
```

Never evaluate on training data. The validation set should be representative of production inputs.

---

## RLHF and Alignment

### Why RLHF Exists

Supervised fine-tuning (SFT) teaches a model to imitate training examples. But "good responses" are easier for humans to recognize than to write. RLHF bridges this gap:

1. **Train a reward model** on human preferences (given two responses, which is better?)
2. **Optimize the LLM** to maximize the reward model's score using PPO (Proximal Policy Optimization)

```
Step 1: Collect preferences
  Prompt → Model generates Response A and Response B
  Human labels: "A is better than B"
  Repeat thousands of times

Step 2: Train reward model
  (prompt, response) → scalar score
  Trained to assign higher scores to human-preferred responses

Step 3: Optimize with PPO
  Model generates response → Reward model scores it → Update model to increase score
  (with KL penalty to prevent diverging too far from the SFT model)
```

### The Alignment Tax

RLHF makes models more helpful and safer, but it has costs:
- Models become more cautious (may refuse edge-case requests that are actually fine)
- Raw capability on benchmarks can decrease slightly
- Training is complex and expensive
- The reward model can be gamed (model learns to produce responses that *look* good to the reward model but aren't actually better)

### Why It Matters for Applied Engineers

You probably won't run RLHF yourself. But understanding it explains:
- Why models sometimes refuse requests (safety training via RLHF)
- Why fine-tuning can bypass safety (you're overwriting RLHF training)
- Why different models have different "personalities" (different RLHF data)
- Why the same base model can produce very different chat models (Llama base vs. Llama Chat)

---

## DPO (Direct Preference Optimization)

### RLHF Is Complex. DPO Simplifies It.

DPO achieves similar results to RLHF without training a separate reward model or using RL.

**Key insight:** Instead of training a reward model and then optimizing against it, DPO directly optimizes the language model on preference pairs.

**Data format:**
```json
{
  "prompt": "Explain quantum computing to a 10-year-old.",
  "chosen": "Imagine you have a magic coin that can be heads AND tails at the same time...",
  "rejected": "Quantum computing leverages superposition of qubits in Hilbert space..."
}
```

**Why DPO is becoming standard:**
- No reward model to train and maintain
- No RL training loop (PPO is notoriously unstable)
- Simpler to implement -- it's just a different loss function on preference pairs
- Competitive results with RLHF in most settings
- Easier to debug (standard supervised training pipeline)

### DPO Variants

- **IPO (Identity Preference Optimization):** Addresses DPO's tendency to overfit on easy preferences
- **KTO (Kahneman-Tversky Optimization):** Works with binary feedback (good/bad) instead of pairwise preferences
- **ORPO (Odds Ratio Preference Optimization):** Combines SFT and preference alignment in a single training step

**For interviews:** Know that DPO exists, why it's simpler than RLHF, and that it works on preference pairs. The variants are bonus knowledge.

---

## Evaluation

### How to Measure Fine-Tuning Success

Never ship a fine-tuned model without rigorous evaluation. "It feels better" is not a metric.

### Quantitative Metrics

| Metric | What It Measures | Best For |
|---|---|---|
| **Perplexity** | How "surprised" the model is by test data | General language quality |
| **BLEU** | N-gram overlap with reference text | Translation, short generation |
| **ROUGE** | Recall-oriented overlap with reference | Summarization |
| **Pass@k** | Fraction of problems solved in k attempts | Code generation |
| **Exact match** | Predicted output == expected output | Classification, extraction |
| **F1 score** | Precision/recall balance | Classification |

### The Evaluation Hierarchy

```
1. Automated metrics (fast, cheap, limited)
   ↓
2. LLM-as-judge (moderate cost, good for style/quality)
   ↓
3. Human evaluation (expensive, gold standard)
```

**Automated metrics** catch regressions quickly but miss nuance. A model might score the same ROUGE but produce less coherent text.

**LLM-as-judge** uses a strong model (GPT-4, Claude) to evaluate outputs. Effective for style, helpfulness, and format compliance. Correlates well with human judgment for many tasks.

**Human evaluation** is the gold standard but expensive. Use it for final validation and to calibrate your automated metrics.

### Evaluation Best Practices

1. **Hold out a test set from the start.** Never evaluate on training data.
2. **Compare to baseline.** Always measure the base model (with good prompting) on the same test set.
3. **Test multiple dimensions.** A model that's better at your task but worse at general coherence might not be a net win.
4. **Check for regression.** Run general-capability benchmarks to ensure you haven't broken other abilities.
5. **Use production-representative data.** Your test set should match what the model will actually see.
6. **Track metrics over training.** Plot validation metrics at checkpoints to pick the best checkpoint, not just the final one.

---

## Cost and Infrastructure

### GPU Requirements by Method

| Model Size | Full Fine-Tune (FP16) | LoRA (FP16) | QLoRA (4-bit) |
|---|---|---|---|
| 1.5B | ~12GB (1x RTX 4090) | ~8GB (1x RTX 4090) | ~4GB (1x RTX 3080) |
| 7B | ~56GB (1x A100-80GB) | ~16GB (1x RTX 4090) | ~8GB (1x RTX 4090) |
| 13B | ~104GB (2x A100-80GB) | ~28GB (1x A100-40GB) | ~12GB (1x RTX 4090) |
| 70B | ~560GB (8x A100-80GB) | ~160GB (2x A100-80GB) | ~40GB (1x A100-80GB) |

These are approximate. Actual VRAM depends on sequence length, batch size, and optimizer.

### Training Time Estimates

| Scenario | Hardware | Time | Cost (cloud) |
|---|---|---|---|
| 7B LoRA, 1K examples, 3 epochs | 1x A100-80GB | ~30 minutes | ~$3-5 |
| 7B LoRA, 10K examples, 3 epochs | 1x A100-80GB | ~3-5 hours | ~$15-30 |
| 7B QLoRA, 10K examples, 3 epochs | 1x RTX 4090 | ~4-8 hours | ~$5-10 (consumer GPU) |
| 70B QLoRA, 10K examples, 3 epochs | 1x A100-80GB | ~24-48 hours | ~$100-200 |
| 70B full fine-tune, 50K examples | 8x A100-80GB | ~1-3 days | ~$5,000-15,000 |

### Training Platforms

**API fine-tuning (managed, simplest):**

| Provider | Models | Cost | Notes |
|---|---|---|---|
| OpenAI | GPT-4o-mini, GPT-4o | ~$3-25/1M training tokens | Simplest workflow, limited control |
| Anthropic | Claude (select models) | Varies | Enterprise offering |
| Google | Gemini | Varies | Vertex AI integration |
| Together AI | Open-source models | ~$1-5/1M training tokens | Good balance of control and ease |

**Cloud GPU training (more control):**

| Provider | GPU Options | Hourly Cost (A100-80GB) | Notes |
|---|---|---|---|
| Lambda Labs | A100, H100 | ~$1.50-3.00/hr | Simple, GPU-focused |
| RunPod | A100, H100 | ~$1.50-2.50/hr | On-demand and spot |
| AWS (SageMaker) | A100, various | ~$5-8/hr | Enterprise features |
| GCP (Vertex AI) | A100, TPU | ~$4-7/hr | TPU option for large runs |

**Local training:**

| GPU | VRAM | Can Fine-Tune | Street Price |
|---|---|---|---|
| RTX 3090 | 24GB | 7B QLoRA, 1.5B LoRA | ~$600-800 used |
| RTX 4090 | 24GB | 7B QLoRA, 1.5B LoRA | ~$1,500-2,000 |
| A6000 | 48GB | 13B QLoRA, 7B LoRA | ~$3,000-4,000 |

### Cost Decision Framework

```
Budget < $100:     QLoRA on consumer GPU or API fine-tuning (small dataset)
Budget $100-1K:    LoRA on cloud GPU, experiment with multiple configs
Budget $1K-10K:    Full LoRA runs on larger models, hyperparameter search
Budget $10K+:      Full fine-tuning, continued pre-training, large-scale experiments
```

---

## Interview Questions and Answers

### Q: "When would you fine-tune versus using RAG?"

**Strong answer:** Fine-tuning and RAG solve different problems. RAG addresses "what does the model know" -- it gives the model access to external knowledge at inference time. Fine-tuning addresses "how does the model behave" -- it changes the model's output style, format, or reasoning patterns. If I need the model to answer questions about our internal documentation, I'd use RAG. If I need the model to always respond in our company's specific JSON schema or match a particular writing style, I'd fine-tune. In practice, many production systems use both: RAG for knowledge retrieval and a fine-tuned model for consistent output formatting.

### Q: "Explain LoRA to a non-ML engineer."

**Strong answer:** When you fine-tune a model normally, you update billions of parameters -- that's expensive and slow. LoRA is based on the observation that the actual changes needed for a specific task are much simpler than the full model. Instead of updating the entire weight matrix, you decompose the update into two small matrices that multiply together. It's like saying "I don't need to rewrite the whole book, I just need to add a few pages of notes." The result is 100x fewer parameters to train, 10x less memory, and comparable quality for most tasks.

### Q: "How would you evaluate a fine-tuned model?"

**Strong answer:** I'd set up a three-layer evaluation system. First, automated metrics on a held-out test set -- exact match for classification, ROUGE for summarization, or task-specific metrics. This catches regressions quickly during development. Second, LLM-as-judge evaluation where I have a strong model rate outputs on dimensions like correctness, style adherence, and completeness -- this captures quality aspects that automated metrics miss. Third, human evaluation on a random sample for final validation. I'd also run the model against general benchmarks to check for catastrophic forgetting. Crucially, I'd always compare against the base model with optimized prompts as a baseline -- if the base model with good prompts matches the fine-tuned model, fine-tuning wasn't worth it.

### Q: "What's the difference between RLHF and DPO?"

**Strong answer:** Both align models with human preferences, but they work differently. RLHF is a three-stage process: collect preference data, train a separate reward model, then optimize the LLM with reinforcement learning (PPO) against that reward model. DPO collapses this into a single step -- it directly optimizes the LLM on preference pairs using a modified loss function, no reward model needed. DPO is simpler to implement, more stable to train, and produces comparable results. The trade-off is that RLHF can be more flexible because the reward model can generalize beyond the training preferences, while DPO is more tightly coupled to the specific preference examples it sees. In practice, DPO has become the standard for teams doing preference alignment because the simplicity wins.

### Q: "Your fine-tuned model performs well on test data but poorly in production. What happened?"

**Strong answer:** This is a distribution mismatch problem. I'd investigate several things: First, does the test set actually represent production inputs? If the test set was carved from the same distribution as the training data but production users write differently, the test metrics are misleading. Second, has the input distribution shifted? Users might be asking things we didn't anticipate. Third, is there a data leakage issue where test examples are too similar to training examples? Fourth, I'd check whether the model is overfitting to patterns in the training data that don't generalize -- maybe it memorized specific formatting instead of learning the underlying task. The fix depends on the diagnosis, but it usually involves collecting production examples, adding them to the test set, and potentially retraining with more diverse data.

### Q: "How would you fine-tune a model on a limited budget?"

**Strong answer:** I'd optimize along three axes. For compute: use QLoRA so I can train on consumer GPUs or cheap cloud instances -- a 7B model with QLoRA needs only 8GB VRAM. For data: generate synthetic training data using a stronger model like GPT-4, which costs dollars instead of the thousands you'd spend on human annotation. For iteration speed: start with a small dataset (200-500 examples), evaluate, and only scale up data collection if metrics show the model is underfitting. I'd also use a smaller model (7B-13B) rather than a 70B unless quality requirements demand it. A well-fine-tuned 7B model often outperforms a prompted 70B model on specific tasks.

---

## Key Takeaways

1. **Fine-tuning changes behavior, RAG adds knowledge.** Choose based on your actual problem.
2. **LoRA is the default choice.** Full fine-tuning is rarely justified for applied work.
3. **Data quality beats data quantity.** 500 excellent examples beat 50,000 mediocre ones.
4. **Always compare against a prompting baseline.** Fine-tuning isn't free -- make sure it's actually better.
5. **Start small, measure, iterate.** Don't commit to a 70B full fine-tune before trying QLoRA on a 7B.
6. **Evaluation is non-negotiable.** Automate it, layer it, and run it on every training run.
7. **Budget for maintenance.** New base models, data drift, and changing requirements mean fine-tuning is ongoing.

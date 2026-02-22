# Fine-Tuning & Model Customization -- Deep Dive

## Model Distillation

### What Is Distillation?

Distillation trains a smaller "student" model to replicate the behavior of a larger "teacher" model. The goal: a smaller model that performs nearly as well as the larger one on your specific tasks, at a fraction of the inference cost.

```
Teacher model (70B, GPT-4, Claude)
    │
    │  Generate high-quality outputs on your task
    ↓
Training data: (input, teacher_output) pairs
    │
    │  Fine-tune smaller model on this data
    ↓
Student model (7B-13B, runs on cheaper hardware)
```

### Why Distillation Works

A large model's outputs contain more information than a simple label:
- **Soft predictions** -- the distribution of probabilities across tokens, not just the most likely token
- **Reasoning patterns** -- chain-of-thought outputs show the student *how* to think
- **Style and nuance** -- the teacher's formatting, tone, and level of detail

A distilled 8B model can outperform a base 70B on specific tasks because:
1. The 70B base model spreads its capacity across everything it learned in pre-training
2. The distilled 8B model focuses entirely on your task
3. The distilled model learned from the 70B's refined outputs, not raw internet text

### Distillation Pipeline

```
Step 1: Define your task scope
  - What inputs will the model see?
  - What outputs do you need?
  - What quality bar must be met?

Step 2: Generate teacher data
  - Run the teacher model on representative inputs
  - Include chain-of-thought when reasoning matters
  - Generate multiple outputs per input, keep the best

Step 3: Filter and curate
  - Score outputs for correctness (automated + spot-check)
  - Remove duplicates and low-quality examples
  - Ensure diversity across input types

Step 4: Train the student
  - Standard fine-tuning (LoRA or QLoRA on the student)
  - Use the teacher's outputs as training targets
  - Train for 1-3 epochs, monitor validation loss

Step 5: Evaluate
  - Compare student vs. teacher on held-out test set
  - If student reaches 90%+ of teacher quality, ship it
  - If not, add more training data or increase student model size
```

### Cost Comparison

| Approach | Inference Cost (per 1M tokens) | Latency | Quality |
|---|---|---|---|
| GPT-4o (teacher) | ~$2.50-10 | 5-15s | Highest |
| Claude Sonnet (teacher) | ~$3-15 | 3-10s | Highest |
| Distilled 8B (self-hosted) | ~$0.10-0.30 | 0.5-2s | 85-95% of teacher |
| Distilled 8B (API) | ~$0.05-0.20 | 0.5-2s | 85-95% of teacher |

At high volume (millions of requests), the cost savings from distillation can be 10-50x.

---

## Synthetic Data Generation

### Why Synthetic Data?

Human annotation is slow (weeks), expensive ($10-50+ per example for expert domains), and hard to scale. Synthetic data generation uses strong LLMs to produce training data at scale.

**Where it shines:**
- Bootstrapping when you have zero labeled data
- Augmenting small datasets to improve coverage
- Generating diverse examples for edge cases
- Creating preference pairs for DPO

**Where it fails:**
- When the teacher model doesn't understand the domain (e.g., highly specialized medical procedures)
- When examples require real-world verification (e.g., "Is this code actually correct?")
- When the synthetic data is too homogeneous (model collapse)

### Synthetic Data Pipeline

```
1. SEED DATA
   Collect 20-50 real examples of ideal inputs and outputs
   These anchor the distribution and set the quality bar

2. GENERATION PROMPTS
   Write diverse prompt templates that instruct the teacher to generate new examples
   Vary: topic, difficulty, length, style, edge cases

3. BATCH GENERATION
   Run the teacher model across all prompt templates
   Generate 5-10x more data than you need

4. QUALITY FILTERING
   Automated checks: format compliance, length, basic correctness
   LLM-as-judge: have a second model rate quality
   Human review: spot-check 5-10% of the data

5. DEDUPLICATION
   Remove exact duplicates
   Remove near-duplicates (embedding similarity > 0.95)

6. DIVERSITY ANALYSIS
   Cluster examples by embedding
   Ensure coverage across all expected input categories
   Generate more data for underrepresented clusters

7. FINAL DATASET
   Format for training (JSONL, chat format)
   Split into train/validation/test
```

### Avoiding Model Collapse

Model collapse occurs when training on synthetic data produces increasingly homogeneous outputs, which then generate even more homogeneous data in the next iteration.

**Prevention strategies:**
- Always mix synthetic data with real data (even a small amount of real data helps)
- Use diverse generation prompts -- vary instructions, personas, and constraints
- Generate from multiple teacher models when possible
- Monitor output diversity metrics (unique n-grams, embedding spread)
- Never train on data from the model you're fine-tuning (use a *different* model as teacher)

### Cost of Synthetic Data Generation

| Dataset Size | Teacher Model | Approximate Cost | Time |
|---|---|---|---|
| 1,000 examples | GPT-4o | $5-20 | 1-2 hours |
| 5,000 examples | GPT-4o | $25-100 | 4-8 hours |
| 10,000 examples | GPT-4o-mini | $5-20 | 2-4 hours |
| 50,000 examples | GPT-4o-mini | $25-100 | 8-24 hours |

These estimates assume ~500 tokens per example (input + output). Real costs depend on token counts and pricing tiers.

---

## Continued Pre-Training

### When Base Fine-Tuning Isn't Enough

Instruction fine-tuning teaches a model to follow instructions in a new way. Continued pre-training teaches a model entirely new knowledge.

**Use continued pre-training when:**
- The domain has specialized vocabulary the base model doesn't know (medical, legal, financial)
- You have a large corpus of unlabeled domain text (millions of tokens)
- You need the model to "think" in the domain's language, not just answer questions about it

**Don't use continued pre-training when:**
- You can get the knowledge into the context via RAG
- You have fewer than a few million tokens of domain text
- The domain is already well-represented in the base model's training data

### How It Works

```
Base Model (trained on internet text)
    ↓
Continued Pre-Training (next-token prediction on domain text)
    ↓
Domain-Adapted Base Model (understands domain vocabulary and patterns)
    ↓
Instruction Fine-Tuning (SFT/LoRA on instruction-following data)
    ↓
Domain-Specialized Chat Model
```

The key: continued pre-training uses the same objective as original pre-training (next-token prediction), not instruction-following. You're extending the model's knowledge, not teaching it to follow instructions.

### Practical Considerations

- **Data volume:** Minimum ~10M tokens of domain text for meaningful adaptation. Ideal is 100M+ tokens.
- **Learning rate:** Much lower than fine-tuning (1e-5 to 5e-5). You're adjusting knowledge, not behavior.
- **Duration:** Can take days to weeks depending on corpus size and compute.
- **Cost:** Significant -- continued pre-training on a 7B model with 100M tokens on A100s runs $500-$2,000+.
- **Forgetting risk:** Continued pre-training can degrade general capabilities. Mix in some general text (5-10% of training data) to mitigate.

### Examples in Practice

| Domain | Data Source | Tokens | Result |
|---|---|---|---|
| Medical | PubMed, clinical notes | 50B+ | BioMistral, PMC-LLaMA |
| Legal | Case law, statutes, contracts | 10B+ | SaulLM |
| Code | GitHub, documentation | 100B+ | CodeLlama, DeepSeek-Coder |
| Finance | SEC filings, earnings calls, news | 50B+ | BloombergGPT |

---

## Model Merging and Composition

### What Is Model Merging?

Model merging combines the weights of multiple fine-tuned models into a single model without additional training. It sounds like it shouldn't work, but it does -- often surprisingly well.

### Merging Methods

**Linear Interpolation (simplest):**
```
W_merged = alpha * W_model_A + (1 - alpha) * W_model_B
```
- Blend two models with a mixing coefficient
- Works when models were fine-tuned from the same base
- Alpha typically between 0.3-0.7

**SLERP (Spherical Linear Interpolation):**
- Interpolates along the surface of a hypersphere instead of a straight line
- Better preserves the magnitude of weight vectors
- Often produces higher quality merges than linear interpolation
- Works best for merging two models

**TIES (TrIm, Elect, Sign & Merge):**
- Identifies conflicting parameters between models
- Resolves conflicts by majority vote on sign direction
- Trims parameters with small magnitudes (noise)
- Handles 3+ models better than SLERP

**DARE (Drop And REscale):**
- Randomly drops a fraction of each model's delta (difference from base)
- Rescales remaining deltas to compensate
- Reduces interference between merged models
- Particularly effective when merging many models

### LoRA Adapter Composition

With LoRA, you can load different adapters at inference time:

```
Base Model (frozen)
    ├── LoRA Adapter: Medical (swap in for medical queries)
    ├── LoRA Adapter: Legal (swap in for legal queries)
    └── LoRA Adapter: Code (swap in for code tasks)
```

**Approaches:**
- **Adapter switching:** Route requests to the appropriate adapter (simplest, no quality loss)
- **Adapter stacking:** Apply multiple adapters simultaneously (mixed results, can degrade quality)
- **Adapter merging:** Merge multiple LoRA adapters into one (permanent, no switching overhead)

### When to Merge vs. Switch vs. Retrain

| Approach | Latency | Quality | Maintenance | Best When |
|---|---|---|---|---|
| Adapter switching | +0ms (preloaded) | Highest | Maintain each adapter | Tasks are clearly separable |
| Model merging | +0ms | Good (80-95%) | One model to maintain | You want simplicity |
| Multi-task fine-tuning | +0ms | Highest | One training run | You have data for all tasks |

---

## Catastrophic Forgetting

### What It Is

Catastrophic forgetting occurs when fine-tuning on a specific task causes the model to lose previously learned general capabilities.

```
Before fine-tuning:
  General knowledge: 90%    Task performance: 60%

After fine-tuning:
  General knowledge: 50%    Task performance: 95%
```

The model gets great at your task but forgets how to do other things.

### Why It Happens

- Neural networks store knowledge across shared parameters
- Fine-tuning overwrites these shared parameters to optimize for the new task
- The more aggressively you fine-tune (high learning rate, many epochs), the worse it gets
- Small, homogeneous datasets amplify the effect

### Mitigation Strategies

**LoRA (primary defense):**
- Freezes base weights, trains only small adapters
- Base knowledge is preserved by design
- The single best protection against catastrophic forgetting

**Replay buffers:**
- Mix general-purpose examples into your fine-tuning data
- Ratio: 5-20% general data + 80-95% task-specific data
- General data can come from existing instruction datasets (Open Orca, Dolly, etc.)

**Lower learning rate:**
- Slower weight updates preserve existing knowledge
- Trade-off: slower convergence, may need more epochs

**Early stopping:**
- Monitor validation loss; stop when it starts increasing
- Later checkpoints often show more forgetting

**Elastic Weight Consolidation (EWC):**
- Penalizes changes to parameters that were important for previous tasks
- More complex to implement but effective for sequential fine-tuning

**Regularization:**
- Weight decay prevents weights from drifting too far
- KL divergence penalty between fine-tuned and base model outputs

### Detection

Run general-capability benchmarks before and after fine-tuning:

| Benchmark | Measures | Quick to Run |
|---|---|---|
| MMLU | General knowledge and reasoning | 30-60 min |
| HellaSwag | Common sense reasoning | 15-30 min |
| ARC | Science reasoning | 15-30 min |
| HumanEval | Code generation | 15-30 min |
| MT-Bench | Multi-turn conversation quality | 30-60 min |

A drop of >5% on these benchmarks after fine-tuning warrants investigation.

---

## Quantization for Deployment

### What Is Quantization?

Quantization reduces the precision of model weights from higher-bit formats (FP32, FP16) to lower-bit formats (INT8, INT4), shrinking model size and increasing inference speed.

```
FP16 weight: 16 bits per parameter → 7B model = 14GB
INT8 weight:  8 bits per parameter → 7B model =  7GB
INT4 weight:  4 bits per parameter → 7B model =  3.5GB
```

### Quantization Methods

| Method | Format | Quality (vs FP16) | Speed | Notes |
|---|---|---|---|---|
| **GPTQ** | INT4, INT3 | 95-99% | Fast (GPU) | Calibration-based, popular for GPU deployment |
| **AWQ** | INT4 | 96-99% | Fast (GPU) | Activation-aware, slightly better quality than GPTQ |
| **GGUF** | Various (Q4_K_M, Q5_K_M, etc.) | 90-99% | Moderate (CPU/GPU) | llama.cpp format, great for CPU inference |
| **bitsandbytes** | INT8, NF4 | 95-99% | Moderate | Used during training (QLoRA), not optimized for inference |
| **SmoothQuant** | INT8 | 98-99% | Fast | Migrates quantization difficulty from activations to weights |

### Quality Impact by Precision

| Precision | Size Reduction | Quality Impact | When to Use |
|---|---|---|---|
| FP16/BF16 | 1x (baseline) | None | Maximum quality, sufficient VRAM |
| INT8 | 2x | Negligible (~1%) | Default for deployment when VRAM is tight |
| INT4 (GPTQ/AWQ) | 4x | Small (~2-5%) | Production deployment on limited hardware |
| INT4 (GGUF Q4_K_M) | 4x | Small (~3-5%) | CPU/laptop deployment |
| INT3 | 5.3x | Moderate (~5-10%) | Extreme memory constraints |
| INT2 | 8x | Significant (~10-20%) | Research only; too much degradation for production |

### Choosing a Quantization Method

```
Deploying on GPU with enough VRAM?
├── Need maximum quality → FP16/BF16
├── Need 2x compression → INT8 (any method)
└── Need 4x compression → AWQ or GPTQ

Deploying on CPU or laptop?
└── GGUF (Q4_K_M is a good default)

Training with QLoRA?
└── bitsandbytes NF4 (built into the training stack)
```

### Quantization After Fine-Tuning

Typical workflow:
1. Fine-tune in FP16/BF16 (or QLoRA for training only)
2. Merge LoRA adapters back into base model
3. Quantize the merged model for deployment
4. Evaluate the quantized model to confirm quality

Do not skip step 4. Quantization can interact poorly with fine-tuning -- especially if the fine-tuning pushed weights into unusual ranges.

---

## Training Infrastructure

### What an Applied Engineer Should Know

You don't need to implement distributed training, but you need to know enough to:
- Choose the right hardware and configuration
- Debug training failures
- Estimate costs and timelines

### DeepSpeed

Microsoft's distributed training library. Relevant for multi-GPU training.

**ZeRO Stages (Zero Redundancy Optimizer):**

| Stage | What It Distributes | Memory Savings | Communication Cost |
|---|---|---|---|
| ZeRO-1 | Optimizer states | ~4x | Low |
| ZeRO-2 | + Gradients | ~8x | Moderate |
| ZeRO-3 | + Parameters | ~Nx (N = # GPUs) | Highest |

- **ZeRO-1/2:** Use when the model fits on one GPU but optimizer states don't
- **ZeRO-3:** Use when the model itself doesn't fit on one GPU
- **ZeRO-Offload:** Offloads to CPU RAM when GPU VRAM is insufficient (slower but cheaper)

### FSDP (Fully Sharded Data Parallel)

PyTorch's native answer to DeepSpeed ZeRO-3. Shards model parameters, gradients, and optimizer states across GPUs.

**When to use FSDP vs DeepSpeed:**
- FSDP: PyTorch native, simpler setup, good for most cases
- DeepSpeed: More features, better documentation, wider community support
- For LoRA on models that fit on one GPU: neither -- just use standard training

### Flash Attention

Not a distributed training technique, but critical for training efficiency:
- 2-4x faster attention computation
- 5-20x less memory for attention
- Enables longer sequence lengths during training
- Essentially free performance -- always use it

### Practical Configuration Guide

```
Single GPU, model fits in VRAM:
  → Standard training with Flash Attention
  → No DeepSpeed/FSDP needed

Single GPU, tight on VRAM:
  → QLoRA + gradient checkpointing
  → Reduce batch size, use gradient accumulation

Multi-GPU, single node (2-8 GPUs):
  → FSDP or DeepSpeed ZeRO-2
  → LoRA usually doesn't need this

Multi-GPU, multi-node:
  → DeepSpeed ZeRO-3
  → Only for full fine-tuning of very large models
  → Consider whether you really need this scale
```

---

## Open-Source Fine-Tuning Stack

### Hugging Face Transformers + PEFT + TRL

The standard stack for fine-tuning open-source models.

**Transformers:** Model loading, tokenization, training loop
**PEFT (Parameter-Efficient Fine-Tuning):** LoRA, QLoRA, prefix tuning, adapters
**TRL (Transformer Reinforcement Learning):** SFT, DPO, RLHF training

```python
# Conceptual workflow (not runnable -- requires GPU)
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# Load model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

# Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
)
model = get_peft_model(model, lora_config)

# Train with SFTTrainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
)
trainer.train()
```

**When to use:** Default choice. Most flexibility, best documentation, largest community.

### Axolotl

Higher-level fine-tuning framework built on top of Transformers/PEFT/TRL.

**What it adds:**
- YAML configuration (no Python code for common workflows)
- Built-in data processing for multiple formats
- Multi-GPU support out of the box
- Predefined configs for common models

```yaml
# axolotl config (simplified)
base_model: meta-llama/Llama-3.1-8B
model_type: LlamaForCausalLM
load_in_4bit: true
adapter: lora
lora_r: 16
lora_alpha: 32
lora_target_modules:
  - q_proj
  - v_proj
  - k_proj
  - o_proj
datasets:
  - path: data/train.jsonl
    type: alpaca
learning_rate: 0.0002
num_epochs: 3
micro_batch_size: 4
gradient_accumulation_steps: 4
```

**When to use:** You want to get a training run going quickly without writing Python. Good for experimentation and standard workflows.

### LLaMA-Factory

Web-UI-based fine-tuning platform for open-source models.

**What it adds:**
- Web interface for configuring and launching training
- Supports 100+ model architectures
- Built-in dataset management
- Real-time training monitoring in the browser

**When to use:** You want the fastest path from "I have data" to "I have a fine-tuned model." Good for teams where not everyone writes Python. Less customizable than raw Transformers/PEFT.

### Comparison

| Feature | Transformers+PEFT+TRL | Axolotl | LLaMA-Factory |
|---|---|---|---|
| Flexibility | Maximum | High | Moderate |
| Ease of use | Moderate | High | Highest |
| Customization | Full Python control | YAML + Python hooks | Web UI + YAML |
| Multi-GPU | Manual config | Built-in | Built-in |
| Community | Largest | Large | Large |
| Best for | Custom pipelines | Standard workflows | Quick experiments |

---

## Safety Considerations

### Fine-Tuning Can Bypass Safety Training

This is the most critical safety concern with fine-tuning. Models like Llama, Mistral, and others go through extensive safety training (RLHF/DPO alignment) before release. Fine-tuning can undo this work.

**How it happens:**
- Fine-tuning data that doesn't include safety-relevant examples lets the model "forget" safety training
- Even a small number of unsafe examples can shift the model's behavior
- Research shows that as few as 100 deliberately harmful examples can remove most safety guardrails from an aligned model

### Responsible Practices

**During data preparation:**
- Audit training data for harmful content, bias, and PII
- Include safety-relevant examples in training data (refusals for harmful requests)
- Filter synthetic data for harmful outputs

**During training:**
- Compare model behavior on safety benchmarks before and after fine-tuning
- Include a held-out safety test set (harmful prompts that should be refused)
- Use DPO with safety-relevant preference pairs

**During deployment:**
- Layer input/output filtering on top of the model
- Monitor for adversarial usage patterns
- Implement rate limiting and logging
- Have a kill switch for rolling back to the base model

### Red-Teaming Fine-Tuned Models

Before deploying a fine-tuned model, test it against adversarial inputs:

1. **Standard safety probes:** Test with known harmful prompt categories (violence, PII extraction, deception)
2. **Jailbreak attempts:** Try common jailbreak patterns to see if fine-tuning made the model more susceptible
3. **Domain-specific risks:** Test for risks specific to your use case (e.g., a medical model giving dangerous advice)
4. **Bias testing:** Check if fine-tuning amplified biases present in the training data

### Legal and Compliance Considerations

- **License restrictions:** Many base models (Llama, Mistral) have licenses that restrict certain uses. Fine-tuning doesn't change the license.
- **Data rights:** Ensure you have the right to use your training data. Synthetic data from APIs may be subject to provider terms.
- **Model outputs:** You're responsible for what your fine-tuned model produces, regardless of what the base model would have done.
- **Regulatory requirements:** Healthcare (HIPAA), finance (SOX), and other regulated industries have specific requirements around AI models.

---

## Advanced Topics

### Mixture of Experts (MoE) and Fine-Tuning

MoE models (like Mixtral) have multiple "expert" sub-networks. During inference, a router selects which experts to activate for each token.

**Fine-tuning MoE models:**
- LoRA can target specific experts or the router
- Training data distribution affects which experts get updated
- MoE models are more parameter-efficient to fine-tune (only active experts need VRAM)

### Speculative Decoding with Fine-Tuned Models

Use a small, fast fine-tuned model as a "draft" model to speed up inference from a larger model:

```
Small fine-tuned model (1B): generates draft tokens quickly
Large model (70B): verifies draft tokens in parallel

Result: 2-3x faster inference with identical output quality
```

This works particularly well when the fine-tuned small model is distilled from the large model on your specific task.

### Continual Learning

Training a model on a sequence of tasks without forgetting previous ones:

```
Task A → Fine-tune → Model_A
Task B → Fine-tune Model_A → Model_AB (but don't forget A)
Task C → Fine-tune Model_AB → Model_ABC (but don't forget A or B)
```

**Approaches:**
- Replay: Mix examples from previous tasks into each new training round
- Regularization: Penalize changes to parameters important for previous tasks
- Architecture: Use separate adapters per task, compose at inference

This matters in production systems where you continuously add capabilities to a model.

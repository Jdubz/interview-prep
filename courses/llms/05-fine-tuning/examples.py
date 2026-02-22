"""
Fine-Tuning & Model Customization -- Complete, Runnable Patterns

These examples cover the parts of the fine-tuning workflow that don't
require GPU hardware: data preparation, quality filtering, evaluation
harnesses, cost estimation, and decision frameworks.

All code is runnable with standard Python (no ML libraries required).
"""

import json
import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# 1. DATA PREPARATION PIPELINE
# ---------------------------------------------------------------------------

# --- Data formats ---

# The three dominant formats for fine-tuning data. Each training framework
# expects a specific format, so conversion between them is a common task.


def to_openai_chat_format(
    instruction: str,
    response: str,
    system_prompt: str = "You are a helpful assistant.",
) -> dict:
    """Convert an instruction/response pair to OpenAI's chat fine-tuning format.

    OpenAI expects JSONL where each line has a 'messages' array with role/content
    pairs. This is the most common format for API-based fine-tuning.
    """
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ]
    }


def to_alpaca_format(instruction: str, response: str, input_text: str = "") -> dict:
    """Convert to Alpaca format, common in open-source fine-tuning.

    Alpaca format separates the instruction (what to do) from the input
    (what to do it on). Many open-source tools accept this natively.
    """
    return {
        "instruction": instruction,
        "input": input_text,
        "output": response,
    }


def to_sharegpt_format(turns: list[tuple[str, str]]) -> dict:
    """Convert multi-turn conversations to ShareGPT format.

    ShareGPT format handles multi-turn conversations where 'human' and 'gpt'
    alternate. Used by tools like Axolotl and LLaMA-Factory.
    """
    conversations = []
    for human_msg, gpt_msg in turns:
        conversations.append({"from": "human", "value": human_msg})
        conversations.append({"from": "gpt", "value": gpt_msg})
    return {"conversations": conversations}


def to_dpo_format(prompt: str, chosen: str, rejected: str) -> dict:
    """Format a preference pair for DPO training.

    DPO needs a prompt with two responses: one the model should learn to
    prefer (chosen) and one it should learn to avoid (rejected).
    """
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }


# --- Format conversion ---


def convert_alpaca_to_openai(
    alpaca_examples: list[dict],
    system_prompt: str = "You are a helpful assistant.",
) -> list[dict]:
    """Convert a dataset from Alpaca format to OpenAI chat format.

    In practice, you often receive data in one format and need another.
    The key difference: Alpaca separates instruction from input, while
    OpenAI merges them into a single user message.
    """
    openai_examples = []
    for ex in alpaca_examples:
        # Alpaca has an optional 'input' field that provides context
        user_content = ex["instruction"]
        if ex.get("input"):
            user_content += f"\n\n{ex['input']}"

        openai_examples.append(
            to_openai_chat_format(user_content, ex["output"], system_prompt)
        )
    return openai_examples


# --- Data cleaning ---


def clean_text(text: str) -> str:
    """Basic text cleaning for training data.

    Garbage in, garbage out. Models learn the exact patterns in training
    data, including artifacts like extra whitespace and encoding errors.
    """
    # Normalize whitespace (models are sensitive to whitespace patterns)
    text = " ".join(text.split())
    # Remove common artifacts from web scraping or copy-paste
    text = text.replace("\x00", "")  # null bytes
    text = text.replace("\ufeff", "")  # BOM
    text = text.replace("\u200b", "")  # zero-width space
    return text.strip()


def clean_example(example: dict) -> dict:
    """Clean all text fields in a training example."""
    cleaned = {}
    for key, value in example.items():
        if isinstance(value, str):
            cleaned[key] = clean_text(value)
        elif isinstance(value, list):
            cleaned[key] = [
                {k: clean_text(v) if isinstance(v, str) else v for k, v in item.items()}
                if isinstance(item, dict)
                else item
                for item in value
            ]
        else:
            cleaned[key] = value
    return cleaned


# --- Deduplication ---


def compute_content_hash(example: dict) -> str:
    """Compute a hash of the text content for deduplication.

    Exact deduplication catches verbatim copies. Near-deduplication
    (embedding similarity) catches paraphrases but requires an
    embedding model, so we handle that separately.
    """
    # Serialize the full structure to catch duplicates regardless of format.
    # Using json.dumps with sort_keys ensures stable ordering.
    content = json.dumps(example, sort_keys=True, ensure_ascii=False).lower()
    return hashlib.md5(content.encode()).hexdigest()


def deduplicate_dataset(examples: list[dict]) -> list[dict]:
    """Remove exact duplicate examples based on content hash.

    Duplicates waste training compute and cause the model to overfit
    on repeated patterns. Even 2-3 duplicates of the same example
    can measurably skew model behavior.
    """
    seen_hashes: set[str] = set()
    unique_examples = []

    for example in examples:
        content_hash = compute_content_hash(example)
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_examples.append(example)

    removed = len(examples) - len(unique_examples)
    print(f"Deduplication: {len(examples)} -> {len(unique_examples)} ({removed} removed)")
    return unique_examples


# --- Validation ---


def validate_openai_format(example: dict) -> list[str]:
    """Validate that an example conforms to OpenAI's chat fine-tuning format.

    API fine-tuning will reject malformed examples, sometimes with
    cryptic errors. Catching issues before uploading saves time.
    """
    errors = []

    if "messages" not in example:
        errors.append("Missing 'messages' field")
        return errors

    messages = example["messages"]
    if not isinstance(messages, list) or len(messages) < 2:
        errors.append("'messages' must be a list with at least 2 entries")
        return errors

    valid_roles = {"system", "user", "assistant"}
    has_user = False
    has_assistant = False

    for i, msg in enumerate(messages):
        if "role" not in msg:
            errors.append(f"Message {i}: missing 'role'")
        elif msg["role"] not in valid_roles:
            errors.append(f"Message {i}: invalid role '{msg['role']}'")
        else:
            if msg["role"] == "user":
                has_user = True
            if msg["role"] == "assistant":
                has_assistant = True

        if "content" not in msg:
            errors.append(f"Message {i}: missing 'content'")
        elif not msg["content"].strip():
            errors.append(f"Message {i}: empty content")

    if not has_user:
        errors.append("No 'user' message found")
    if not has_assistant:
        errors.append("No 'assistant' message found (nothing for the model to learn)")

    return errors


def validate_dataset(examples: list[dict], format_type: str = "openai") -> dict:
    """Validate an entire dataset, returning a summary of issues.

    Always validate before training. A single malformed example can
    cause a training run to fail hours in, wasting time and money.
    """
    validators = {
        "openai": validate_openai_format,
    }
    validate_fn = validators.get(format_type)
    if not validate_fn:
        return {"error": f"Unknown format: {format_type}"}

    total = len(examples)
    valid_count = 0
    error_summary: dict[str, int] = {}

    for i, example in enumerate(examples):
        errors = validate_fn(example)
        if not errors:
            valid_count += 1
        else:
            for error in errors:
                error_summary[error] = error_summary.get(error, 0) + 1

    return {
        "total_examples": total,
        "valid": valid_count,
        "invalid": total - valid_count,
        "validity_rate": valid_count / total if total > 0 else 0,
        "error_breakdown": error_summary,
    }


# --- Train/validation/test split ---


def split_dataset(
    examples: list[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Split a dataset into train/validation/test sets.

    The test set should be touched only once -- for final evaluation.
    Use the validation set for all intermediate evaluation and
    hyperparameter tuning decisions.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    splits = {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }

    for name, split in splits.items():
        print(f"  {name}: {len(split)} examples")

    return splits


# --- Full pipeline ---


def prepare_dataset(
    raw_examples: list[dict],
    system_prompt: str = "You are a helpful assistant.",
) -> dict[str, list[dict]]:
    """End-to-end data preparation pipeline.

    Runs cleaning, format conversion, deduplication, validation,
    and splitting in the correct order. This is the workflow you'd
    run before submitting data for training.
    """
    print(f"Starting with {len(raw_examples)} raw examples")

    # Step 1: Clean
    cleaned = [clean_example(ex) for ex in raw_examples]
    print(f"Cleaned {len(cleaned)} examples")

    # Step 2: Convert to training format
    formatted = convert_alpaca_to_openai(cleaned, system_prompt)
    print(f"Converted to OpenAI chat format")

    # Step 3: Deduplicate
    deduped = deduplicate_dataset(formatted)

    # Step 4: Validate
    validation = validate_dataset(deduped, "openai")
    print(f"Validation: {validation['valid']}/{validation['total_examples']} valid")
    if validation["error_breakdown"]:
        print(f"  Errors: {validation['error_breakdown']}")

    # Remove invalid examples
    valid_examples = []
    for ex in deduped:
        if not validate_openai_format(ex):
            valid_examples.append(ex)

    # Step 5: Split
    print("Splitting dataset:")
    splits = split_dataset(valid_examples)

    return splits


# ---------------------------------------------------------------------------
# 2. DATA QUALITY SCORING
# ---------------------------------------------------------------------------


@dataclass
class QualityScore:
    """Score representing the quality of a single training example.

    Each dimension is scored 0.0-1.0. The overall score is a weighted
    average. Thresholds are configurable -- start strict and relax
    only if you need more data.
    """

    completeness: float = 0.0  # Is the response complete and not truncated?
    length_appropriate: float = 0.0  # Is the response length reasonable for the task?
    format_compliance: float = 0.0  # Does it match the expected format?
    instruction_following: float = 0.0  # Does the response address the instruction?
    overall: float = 0.0


def score_example_quality(
    instruction: str,
    response: str,
    min_response_length: int = 20,
    max_response_length: int = 2000,
    required_format_markers: list[str] | None = None,
) -> QualityScore:
    """Score the quality of a training example using heuristic checks.

    This catches obvious quality issues without needing an LLM judge.
    For deeper quality assessment, use LLM-as-judge (see synthetic
    data generation section).
    """
    score = QualityScore()

    # Completeness: check for truncation signals
    truncation_signals = ["...", "[truncated]", "[continued]", "etc."]
    ends_mid_sentence = response and response[-1] not in ".!?\"')]}"
    has_truncation = any(response.rstrip().endswith(sig) for sig in truncation_signals)
    score.completeness = 0.0 if has_truncation else (0.5 if ends_mid_sentence else 1.0)

    # Length appropriateness
    resp_len = len(response.split())
    if resp_len < min_response_length:
        score.length_appropriate = resp_len / min_response_length
    elif resp_len > max_response_length:
        score.length_appropriate = max(0, 1.0 - (resp_len - max_response_length) / max_response_length)
    else:
        score.length_appropriate = 1.0

    # Format compliance (if specific markers are required)
    if required_format_markers:
        matched = sum(1 for marker in required_format_markers if marker in response)
        score.format_compliance = matched / len(required_format_markers)
    else:
        score.format_compliance = 1.0  # No format requirements

    # Instruction following (basic heuristic: response should relate to instruction)
    instruction_words = set(instruction.lower().split())
    response_words = set(response.lower().split())
    # Check for minimal keyword overlap (very basic check)
    overlap = len(instruction_words & response_words)
    score.instruction_following = min(1.0, overlap / max(1, len(instruction_words) * 0.3))

    # Overall weighted score
    score.overall = (
        0.3 * score.completeness
        + 0.2 * score.length_appropriate
        + 0.2 * score.format_compliance
        + 0.3 * score.instruction_following
    )

    return score


def filter_by_quality(
    examples: list[dict],
    min_score: float = 0.7,
) -> tuple[list[dict], list[dict]]:
    """Filter examples by quality score, returning (kept, rejected).

    Setting the threshold too high leaves you with too little data.
    Setting it too low lets garbage through. Start at 0.7 and adjust
    based on your training results.
    """
    kept = []
    rejected = []

    for ex in examples:
        # Extract instruction and response regardless of format
        instruction = ex.get("instruction", "")
        response = ex.get("output", "")

        # Handle OpenAI format
        if "messages" in ex:
            for msg in ex["messages"]:
                if msg["role"] == "user":
                    instruction = msg["content"]
                if msg["role"] == "assistant":
                    response = msg["content"]

        score = score_example_quality(instruction, response)
        if score.overall >= min_score:
            kept.append(ex)
        else:
            rejected.append(ex)

    print(f"Quality filter: {len(kept)} kept, {len(rejected)} rejected "
          f"(threshold={min_score})")
    return kept, rejected


# ---------------------------------------------------------------------------
# 3. SYNTHETIC DATA GENERATION PIPELINE
# ---------------------------------------------------------------------------

# Prompt templates for generating training data with a teacher model.
# In production, you'd call an LLM API with these prompts. Here we
# define the pipeline structure and prompts.

SYNTHETIC_GENERATION_PROMPTS = {
    "classification": """Generate {count} diverse examples of {task_description}.

For each example, provide:
1. The input text (realistic, varied in length and style)
2. The correct classification label from: {labels}
3. A brief explanation of why this label is correct

Format each example as JSON:
{{"input": "...", "label": "...", "explanation": "..."}}

Requirements:
- Vary the writing style (formal, casual, technical, conversational)
- Include edge cases and ambiguous examples
- Balance the distribution across labels
- Make inputs realistic, not synthetic-sounding""",

    "instruction_following": """Generate {count} diverse instruction-response pairs for {task_description}.

For each pair, provide:
1. A realistic user instruction (varied complexity, phrasing, context)
2. A high-quality response that follows the instruction precisely

Format each as JSON:
{{"instruction": "...", "response": "..."}}

Requirements:
- Vary instruction complexity from simple to multi-step
- Responses should be complete and well-structured
- Include edge cases (ambiguous instructions, missing context)
- Match the tone and style expected in {domain}""",

    "preference_pairs": """Generate {count} preference pairs for {task_description}.

For each pair, provide:
1. A realistic user prompt
2. A GOOD response (helpful, accurate, well-structured)
3. A WORSE response (has specific, identifiable flaws)

Format each as JSON:
{{"prompt": "...", "chosen": "...", "rejected": "..."}}

The rejected response should have realistic flaws:
- Too verbose or too terse
- Missing key information
- Incorrect formatting
- Slightly off-topic
- Overly generic

Do NOT make the rejected response obviously terrible -- it should be
subtly worse, as this is what the model needs to learn to distinguish.""",
}


@dataclass
class SyntheticDataConfig:
    """Configuration for synthetic data generation.

    Separating config from execution makes it easy to reproduce
    experiments and adjust parameters between runs.
    """

    task_description: str
    domain: str = "general"
    count_per_prompt: int = 10
    num_prompt_variations: int = 5
    labels: list[str] = field(default_factory=list)
    template_type: str = "instruction_following"
    teacher_model: str = "gpt-4o"
    temperature: float = 0.8  # Higher for diversity in synthetic data
    quality_threshold: float = 0.7


def build_generation_prompts(config: SyntheticDataConfig) -> list[str]:
    """Build a set of diverse generation prompts from a config.

    Using multiple prompt variations is critical for synthetic data
    diversity. A single prompt template produces homogeneous data.
    """
    template = SYNTHETIC_GENERATION_PROMPTS.get(config.template_type, "")
    if not template:
        raise ValueError(f"Unknown template type: {config.template_type}")

    prompts = []
    # Generate variations by adjusting count and adding diversity hints
    diversity_hints = [
        "Focus on common, everyday scenarios.",
        "Focus on edge cases and unusual situations.",
        "Focus on professional/business contexts.",
        "Focus on technical and specialized contexts.",
        "Focus on casual and conversational contexts.",
        "Focus on examples where the correct answer is ambiguous or requires nuance.",
        "Focus on very short inputs (1-2 sentences).",
        "Focus on longer, detailed inputs (paragraph-length).",
    ]

    for i in range(config.num_prompt_variations):
        hint = diversity_hints[i % len(diversity_hints)]
        prompt = template.format(
            count=config.count_per_prompt,
            task_description=config.task_description,
            labels=", ".join(config.labels) if config.labels else "N/A",
            domain=config.domain,
        )
        prompt += f"\n\nDiversity focus for this batch: {hint}"
        prompts.append(prompt)

    return prompts


def estimate_synthetic_data_cost(
    config: SyntheticDataConfig,
    avg_tokens_per_example: int = 500,
) -> dict:
    """Estimate the cost of generating synthetic data.

    These estimates help you budget before committing to a generation
    run. Actual costs depend on the teacher model's pricing.
    """
    total_examples = config.count_per_prompt * config.num_prompt_variations
    total_tokens = total_examples * avg_tokens_per_example

    # Approximate pricing per 1M tokens (input + output combined)
    pricing = {
        "gpt-4o": 7.50,      # ~$2.50 input + ~$10 output, blended
        "gpt-4o-mini": 0.60, # ~$0.15 input + ~$0.60 output, blended
        "claude-sonnet": 6.00,
    }

    cost_per_million = pricing.get(config.teacher_model, 5.0)
    estimated_cost = (total_tokens / 1_000_000) * cost_per_million

    return {
        "total_examples": total_examples,
        "total_tokens": total_tokens,
        "estimated_cost_usd": round(estimated_cost, 2),
        "teacher_model": config.teacher_model,
        "examples_after_filtering": int(total_examples * config.quality_threshold),
    }


# ---------------------------------------------------------------------------
# 4. TRAINING CONFIGURATION BUILDER
# ---------------------------------------------------------------------------


@dataclass
class LoRAConfig:
    """LoRA training configuration.

    These defaults are reasonable starting points for most tasks.
    Adjust based on your specific requirements -- see the cheat sheet
    for guidance on each parameter.
    """

    rank: int = 16
    alpha: int = 32  # Typically 2x rank
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    dropout: float = 0.05
    bias: str = "none"


@dataclass
class TrainingConfig:
    """Complete training configuration.

    Captures all the decisions you need to make before launching a
    training run. Use build_training_config() below to generate
    a config based on your constraints.
    """

    # Model
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    method: str = "lora"  # "lora", "qlora", "full"

    # LoRA
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    # Training
    learning_rate: float = 2e-4
    num_epochs: int = 3
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    lr_scheduler: str = "cosine"
    max_seq_length: int = 2048

    # Quantization (for QLoRA)
    load_in_4bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"

    # Infrastructure
    bf16: bool = True
    gradient_checkpointing: bool = True


def build_training_config(
    model_size_b: float,
    dataset_size: int,
    gpu_vram_gb: int,
    task_complexity: str = "moderate",
) -> TrainingConfig:
    """Generate a training configuration based on constraints.

    This encodes the practical knowledge of what works for different
    combinations of model size, data size, and hardware.
    """
    config = TrainingConfig()

    # Choose method based on GPU VRAM vs model size
    # Rule of thumb: FP16 model needs ~2x parameter count in GB
    model_fp16_gb = model_size_b * 2
    model_4bit_gb = model_size_b * 0.5

    if gpu_vram_gb >= model_fp16_gb * 3:
        # Enough for full fine-tuning (model + gradients + optimizer)
        config.method = "full"
        config.learning_rate = 2e-5
    elif gpu_vram_gb >= model_fp16_gb * 1.2:
        # Enough for LoRA in FP16
        config.method = "lora"
        config.learning_rate = 2e-4
    elif gpu_vram_gb >= model_4bit_gb * 1.5:
        # Need QLoRA
        config.method = "qlora"
        config.load_in_4bit = True
        config.learning_rate = 2e-4
    else:
        # Won't fit -- recommend smaller model
        print(f"WARNING: {model_size_b}B model unlikely to fit in {gpu_vram_gb}GB VRAM "
              f"even with QLoRA. Consider a smaller model.")
        config.method = "qlora"
        config.load_in_4bit = True

    # Adjust LoRA rank based on task complexity
    complexity_to_rank = {
        "simple": 8,
        "moderate": 16,
        "complex": 32,
        "very_complex": 64,
    }
    rank = complexity_to_rank.get(task_complexity, 16)
    config.lora = LoRAConfig(rank=rank, alpha=rank * 2)

    # Adjust epochs based on dataset size
    # Fewer epochs for large datasets (less risk of overfitting)
    if dataset_size < 500:
        config.num_epochs = 5
    elif dataset_size < 5000:
        config.num_epochs = 3
    else:
        config.num_epochs = 1

    # Adjust batch size based on available VRAM
    if gpu_vram_gb >= 80:
        config.per_device_batch_size = 8
    elif gpu_vram_gb >= 40:
        config.per_device_batch_size = 4
    else:
        config.per_device_batch_size = 2
        config.gradient_checkpointing = True  # Save VRAM

    return config


def config_to_dict(config: TrainingConfig) -> dict:
    """Convert config to a dictionary suitable for logging or serialization."""
    return {
        "base_model": config.base_model,
        "method": config.method,
        "lora_rank": config.lora.rank,
        "lora_alpha": config.lora.alpha,
        "lora_target_modules": config.lora.target_modules,
        "lora_dropout": config.lora.dropout,
        "learning_rate": config.learning_rate,
        "num_epochs": config.num_epochs,
        "per_device_batch_size": config.per_device_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "effective_batch_size": config.per_device_batch_size * config.gradient_accumulation_steps,
        "warmup_ratio": config.warmup_ratio,
        "max_seq_length": config.max_seq_length,
        "load_in_4bit": config.load_in_4bit,
        "bf16": config.bf16,
        "gradient_checkpointing": config.gradient_checkpointing,
    }


# ---------------------------------------------------------------------------
# 5. EVALUATION HARNESS
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    """Result of evaluating a single example."""

    input_text: str
    expected: str
    predicted: str
    exact_match: bool
    scores: dict[str, float] = field(default_factory=dict)


def compute_exact_match(predicted: str, expected: str) -> bool:
    """Check if prediction exactly matches expected output.

    Used for classification, extraction, and other tasks with
    deterministic correct answers.
    """
    return predicted.strip().lower() == expected.strip().lower()


def compute_rouge_l(predicted: str, expected: str) -> float:
    """Compute ROUGE-L score (longest common subsequence).

    ROUGE-L measures the longest common subsequence between prediction
    and reference, normalized by reference length. Good for summarization
    and open-ended generation where exact match is too strict.
    """
    pred_tokens = predicted.lower().split()
    ref_tokens = expected.lower().split()

    if not ref_tokens or not pred_tokens:
        return 0.0

    # Dynamic programming for LCS length
    m, n = len(pred_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_length = dp[m][n]

    # F1-style combination of precision and recall
    precision = lcs_length / m if m > 0 else 0
    recall = lcs_length / n if n > 0 else 0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_bleu_1(predicted: str, expected: str) -> float:
    """Compute unigram BLEU score (simplified).

    BLEU measures n-gram precision of the prediction against the
    reference. Unigram BLEU is the simplest variant -- it checks
    what fraction of prediction tokens appear in the reference.
    """
    pred_tokens = predicted.lower().split()
    ref_tokens = expected.lower().split()

    if not pred_tokens:
        return 0.0

    ref_counts: dict[str, int] = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1

    matches = 0
    for token in pred_tokens:
        if ref_counts.get(token, 0) > 0:
            matches += 1
            ref_counts[token] -= 1

    precision = matches / len(pred_tokens)

    # Brevity penalty (penalize very short predictions)
    if len(pred_tokens) < len(ref_tokens):
        bp = math.exp(1 - len(ref_tokens) / len(pred_tokens))
    else:
        bp = 1.0

    return precision * bp


def evaluate_predictions(
    test_examples: list[dict[str, str]],
    predictions: list[str],
    metrics: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate a list of predictions against expected outputs.

    This is the core evaluation function. In practice, 'predictions'
    come from running the fine-tuned model on test inputs. Here we
    accept predictions as strings so the harness is model-agnostic.
    """
    if metrics is None:
        metrics = ["exact_match", "rouge_l"]

    assert len(test_examples) == len(predictions), "Predictions must match test examples"

    results: list[EvalResult] = []
    metric_totals: dict[str, float] = {m: 0.0 for m in metrics}

    for example, predicted in zip(test_examples, predictions):
        expected = example.get("expected", example.get("output", ""))
        input_text = example.get("input", example.get("instruction", ""))

        scores: dict[str, float] = {}

        if "exact_match" in metrics:
            em = compute_exact_match(predicted, expected)
            scores["exact_match"] = 1.0 if em else 0.0

        if "rouge_l" in metrics:
            scores["rouge_l"] = compute_rouge_l(predicted, expected)

        if "bleu_1" in metrics:
            scores["bleu_1"] = compute_bleu_1(predicted, expected)

        result = EvalResult(
            input_text=input_text,
            expected=expected,
            predicted=predicted,
            exact_match=compute_exact_match(predicted, expected),
            scores=scores,
        )
        results.append(result)

        for metric, value in scores.items():
            metric_totals[metric] += value

    n = len(results)
    avg_metrics = {m: round(v / n, 4) for m, v in metric_totals.items()}

    return {
        "num_examples": n,
        "metrics": avg_metrics,
        "results": results,
    }


def compare_models(
    baseline_results: dict[str, Any],
    finetuned_results: dict[str, Any],
) -> dict:
    """Compare fine-tuned model against baseline.

    Always compare against a baseline. If the fine-tuned model doesn't
    meaningfully outperform the base model with good prompting, the
    fine-tuning wasn't worth the cost.
    """
    comparison = {}
    for metric in baseline_results["metrics"]:
        baseline_val = baseline_results["metrics"][metric]
        finetuned_val = finetuned_results["metrics"][metric]
        delta = finetuned_val - baseline_val
        relative_improvement = (delta / baseline_val * 100) if baseline_val > 0 else float("inf")

        comparison[metric] = {
            "baseline": baseline_val,
            "fine_tuned": finetuned_val,
            "absolute_delta": round(delta, 4),
            "relative_improvement_pct": round(relative_improvement, 1),
            "worth_it": delta > 0.05,  # >5% improvement threshold
        }

    return comparison


# ---------------------------------------------------------------------------
# 6. FINE-TUNING DECISION ENGINE
# ---------------------------------------------------------------------------


@dataclass
class UseCase:
    """Description of a use case for the decision engine."""

    description: str
    needs_external_knowledge: bool = False
    knowledge_changes_frequently: bool = False
    needs_behavior_change: bool = False  # Style, format, reasoning
    needs_cost_optimization: bool = False
    data_available: int = 0  # Number of labeled examples
    budget_usd: float = 0
    latency_requirement_ms: int = 5000
    quality_requirement: str = "high"  # "low", "moderate", "high", "maximum"


def recommend_approach(use_case: UseCase) -> dict:
    """Recommend a customization approach based on use case constraints.

    This encodes the decision framework from the README as executable
    logic. In an interview, walking through this decision tree
    demonstrates practical engineering judgment.
    """
    recommendations = []
    reasoning = []

    # Step 1: Can prompt engineering handle it?
    if not use_case.needs_behavior_change and not use_case.needs_external_knowledge:
        recommendations.append("prompt_engineering")
        reasoning.append(
            "No specialized knowledge or behavior changes needed. "
            "Start with prompt engineering -- it's the cheapest and fastest to iterate."
        )
        return {
            "primary": "prompt_engineering",
            "alternatives": [],
            "reasoning": reasoning,
            "estimated_cost": "$0 (just engineering time)",
            "time_to_deploy": "Hours to days",
        }

    # Step 2: Does it need external knowledge?
    if use_case.needs_external_knowledge:
        if use_case.knowledge_changes_frequently:
            recommendations.append("rag")
            reasoning.append(
                "Knowledge changes frequently. RAG keeps information fresh "
                "without retraining. Fine-tuning would bake in stale facts."
            )
        else:
            recommendations.append("rag")
            reasoning.append(
                "Needs external knowledge. RAG is the standard approach. "
                "Consider long-context prompting if the corpus is small enough."
            )

    # Step 3: Does it need behavior changes?
    if use_case.needs_behavior_change:
        if use_case.data_available >= 200:
            recommendations.append("fine_tuning")
            reasoning.append(
                f"Behavior change needed with {use_case.data_available} examples available. "
                f"Fine-tuning is appropriate."
            )
        elif use_case.data_available > 0:
            recommendations.append("fine_tuning_with_synthetic_data")
            reasoning.append(
                f"Only {use_case.data_available} examples available. Generate synthetic "
                f"data with a stronger model to reach 200+ examples, then fine-tune."
            )
        else:
            recommendations.append("prompt_engineering")
            reasoning.append(
                "Behavior change needed but no training data available. "
                "Start with prompt engineering while collecting data for future fine-tuning."
            )

    # Step 4: Cost optimization?
    if use_case.needs_cost_optimization:
        recommendations.append("distillation")
        reasoning.append(
            "Cost optimization needed. Distill a large model into a smaller one "
            "for cheaper inference at production scale."
        )

    # Determine primary recommendation
    primary = recommendations[0] if recommendations else "prompt_engineering"
    alternatives = recommendations[1:] if len(recommendations) > 1 else []

    # Estimate cost based on approach
    cost_estimates = {
        "prompt_engineering": "$0 (engineering time only)",
        "rag": "$100-1,000 (embedding pipeline + vector DB)",
        "fine_tuning": f"${max(50, use_case.data_available * 0.05):.0f}-${max(500, use_case.data_available * 0.5):.0f}",
        "fine_tuning_with_synthetic_data": "$200-2,000 (synthetic generation + training)",
        "distillation": "$500-5,000 (teacher inference + student training)",
    }

    return {
        "primary": primary,
        "alternatives": alternatives,
        "reasoning": reasoning,
        "estimated_cost": cost_estimates.get(primary, "Varies"),
        "time_to_deploy": "Days to weeks" if "fine_tuning" in primary else "Hours to days",
    }


# ---------------------------------------------------------------------------
# 7. COST ESTIMATOR
# ---------------------------------------------------------------------------


def estimate_training_cost(
    model_size_b: float,
    dataset_size: int,
    method: str = "lora",
    num_epochs: int = 3,
    avg_seq_length: int = 1024,
    gpu_type: str = "a100_80gb",
) -> dict:
    """Estimate the cost of a fine-tuning run.

    These are rough estimates based on typical throughput numbers.
    Actual costs depend on sequence length, batch size, and provider
    pricing -- treat these as order-of-magnitude estimates for
    planning purposes.
    """
    # Throughput estimates (tokens/second per GPU)
    throughput = {
        "full": {
            "a100_80gb": 3000,
            "a100_40gb": 2000,
            "h100_80gb": 6000,
        },
        "lora": {
            "a100_80gb": 5000,
            "a100_40gb": 3500,
            "h100_80gb": 10000,
            "rtx_4090": 2000,
        },
        "qlora": {
            "a100_80gb": 4000,
            "a100_40gb": 3000,
            "h100_80gb": 8000,
            "rtx_4090": 1500,
            "rtx_3090": 1000,
        },
    }

    # GPU pricing ($/hour, approximate cloud rates)
    gpu_pricing = {
        "a100_80gb": 2.50,
        "a100_40gb": 1.80,
        "h100_80gb": 4.00,
        "rtx_4090": 0.50,
        "rtx_3090": 0.35,
    }

    # Number of GPUs needed
    gpu_count = {
        "full": max(1, math.ceil(model_size_b * 2 * 3 / {"a100_80gb": 80, "a100_40gb": 40, "h100_80gb": 80, "rtx_4090": 24, "rtx_3090": 24}.get(gpu_type, 80))),
        "lora": max(1, math.ceil(model_size_b * 2 / {"a100_80gb": 80, "a100_40gb": 40, "h100_80gb": 80, "rtx_4090": 24, "rtx_3090": 24}.get(gpu_type, 80))),
        "qlora": 1,  # QLoRA is designed for single-GPU
    }

    tokens_per_second = throughput.get(method, {}).get(gpu_type, 2000)
    total_tokens = dataset_size * avg_seq_length * num_epochs
    training_seconds = total_tokens / tokens_per_second
    training_hours = training_seconds / 3600

    num_gpus = gpu_count.get(method, 1)
    hourly_cost = gpu_pricing.get(gpu_type, 2.50) * num_gpus
    total_cost = hourly_cost * training_hours

    return {
        "model_size_b": model_size_b,
        "method": method,
        "dataset_size": dataset_size,
        "num_epochs": num_epochs,
        "total_tokens": total_tokens,
        "gpu_type": gpu_type,
        "num_gpus": num_gpus,
        "estimated_hours": round(training_hours, 1),
        "hourly_cost": round(hourly_cost, 2),
        "estimated_total_cost": round(total_cost, 2),
    }


# ---------------------------------------------------------------------------
# DEMO: Run all pipeline stages on sample data
# ---------------------------------------------------------------------------


def main():
    """Demonstrate the full fine-tuning preparation workflow."""

    print("=" * 60)
    print("FINE-TUNING PREPARATION PIPELINE DEMO")
    print("=" * 60)

    # --- Sample data ---
    raw_data = [
        {
            "instruction": "Classify the sentiment of this customer review.",
            "input": "The product quality is excellent and shipping was fast.",
            "output": "Positive",
        },
        {
            "instruction": "Classify the sentiment of this customer review.",
            "input": "Terrible experience. Product broke after one day.",
            "output": "Negative",
        },
        {
            "instruction": "Classify the sentiment of this customer review.",
            "input": "It's okay. Nothing special but works as described.",
            "output": "Neutral",
        },
        {
            "instruction": "Classify the sentiment of this customer review.",
            "input": "  Amazing  value  for the price!! Highly recommend.  ",
            "output": "Positive",
        },
        # Duplicate (will be caught by dedup)
        {
            "instruction": "Classify the sentiment of this customer review.",
            "input": "The product quality is excellent and shipping was fast.",
            "output": "Positive",
        },
    ]

    # --- 1. Data preparation ---
    print("\n--- DATA PREPARATION ---")
    splits = prepare_dataset(raw_data, system_prompt="You are a sentiment classifier.")

    # --- 2. Quality scoring ---
    print("\n--- QUALITY SCORING ---")
    for example in raw_data[:3]:
        score = score_example_quality(
            example["instruction"] + " " + example.get("input", ""),
            example["output"],
            min_response_length=1,  # Sentiment labels are short
        )
        print(f"  Input: {example['input'][:50]}...")
        print(f"  Score: {score.overall:.2f} (completeness={score.completeness:.2f})")

    # --- 3. Synthetic data config ---
    print("\n--- SYNTHETIC DATA PLANNING ---")
    synth_config = SyntheticDataConfig(
        task_description="customer review sentiment classification",
        domain="e-commerce",
        count_per_prompt=20,
        num_prompt_variations=5,
        labels=["Positive", "Negative", "Neutral"],
        template_type="classification",
        teacher_model="gpt-4o-mini",
    )
    cost_estimate = estimate_synthetic_data_cost(synth_config)
    print(f"  Synthetic data plan:")
    print(f"    Total examples: {cost_estimate['total_examples']}")
    print(f"    Estimated cost: ${cost_estimate['estimated_cost_usd']}")
    print(f"    After filtering: ~{cost_estimate['examples_after_filtering']} examples")

    # --- 4. Training config ---
    print("\n--- TRAINING CONFIGURATION ---")
    config = build_training_config(
        model_size_b=7,
        dataset_size=1000,
        gpu_vram_gb=24,
        task_complexity="simple",
    )
    print(f"  Method: {config.method}")
    print(f"  LoRA rank: {config.lora.rank}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  4-bit quantization: {config.load_in_4bit}")

    # --- 5. Evaluation ---
    print("\n--- EVALUATION ---")
    test_data = [
        {"input": "Great product!", "expected": "Positive"},
        {"input": "Worst purchase ever.", "expected": "Negative"},
        {"input": "It works fine.", "expected": "Neutral"},
    ]
    # Simulated predictions (in practice, these come from the model)
    baseline_preds = ["Positive", "Negative", "Positive"]  # Baseline gets one wrong
    finetuned_preds = ["Positive", "Negative", "Neutral"]  # Fine-tuned gets all right

    baseline_eval = evaluate_predictions(test_data, baseline_preds, ["exact_match"])
    finetuned_eval = evaluate_predictions(test_data, finetuned_preds, ["exact_match"])

    comparison = compare_models(baseline_eval, finetuned_eval)
    print(f"  Baseline exact match:    {baseline_eval['metrics']['exact_match']:.2%}")
    print(f"  Fine-tuned exact match:  {finetuned_eval['metrics']['exact_match']:.2%}")
    print(f"  Improvement: {comparison['exact_match']['relative_improvement_pct']}%")
    print(f"  Worth it: {comparison['exact_match']['worth_it']}")

    # --- 6. Decision engine ---
    print("\n--- DECISION ENGINE ---")
    use_case = UseCase(
        description="Customer support chatbot that responds in our brand voice",
        needs_external_knowledge=True,
        knowledge_changes_frequently=True,
        needs_behavior_change=True,
        data_available=500,
        budget_usd=2000,
    )
    recommendation = recommend_approach(use_case)
    print(f"  Primary recommendation: {recommendation['primary']}")
    for reason in recommendation["reasoning"]:
        print(f"    - {reason}")
    print(f"  Estimated cost: {recommendation['estimated_cost']}")

    # --- 7. Cost estimation ---
    print("\n--- COST ESTIMATION ---")
    cost = estimate_training_cost(
        model_size_b=7,
        dataset_size=1000,
        method="qlora",
        num_epochs=3,
        gpu_type="rtx_4090",
    )
    print(f"  Model: {cost['model_size_b']}B ({cost['method']})")
    print(f"  GPU: {cost['gpu_type']} x{cost['num_gpus']}")
    print(f"  Training time: ~{cost['estimated_hours']} hours")
    print(f"  Estimated cost: ${cost['estimated_total_cost']}")


if __name__ == "__main__":
    main()

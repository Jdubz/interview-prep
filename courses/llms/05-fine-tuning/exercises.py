"""
Fine-Tuning & Model Customization -- Exercises

Complete the TODO sections. Each exercise tests a different aspect of the
fine-tuning workflow that comes up in interviews and real production work.

Run with: python exercises.py
Each exercise has a verify() function that checks your implementation.
"""

import json
import math
import random
from dataclasses import dataclass, field
from typing import Any


# ============================================================================
# EXERCISE 1: Approach Selection
#
# Given a use case description and constraints, determine the right approach:
# prompt engineering, RAG, fine-tuning, or distillation. Provide reasoning.
#
# This is the most common fine-tuning interview question. Interviewers want
# to see that you don't reach for fine-tuning as a default -- you consider
# simpler alternatives first and justify the added complexity.
# ============================================================================


@dataclass
class Scenario:
    """A use case scenario for approach selection."""

    description: str
    needs_private_knowledge: bool
    knowledge_update_frequency: str  # "static", "weekly", "daily", "realtime"
    needs_custom_output_format: bool
    needs_custom_tone_or_style: bool
    labeled_examples_available: int
    monthly_request_volume: int
    max_latency_ms: int
    budget_usd: float


def select_approach(scenario: Scenario) -> dict[str, Any]:
    """Determine the best approach for the given scenario.

    Return a dict with:
    - "approach": one of "prompt_engineering", "rag", "fine_tuning",
      "rag_plus_fine_tuning", "distillation"
    - "reasoning": list of strings explaining the decision
    - "data_strategy": how to get/augment training data (if fine-tuning)
    - "estimated_monthly_cost": rough estimate in USD

    Walk through the decision tree:
    1. Can prompt engineering alone solve it?
    2. Does it need external knowledge? -> RAG
    3. Does it need behavior changes? -> Fine-tuning
    4. Does it need both? -> RAG + fine-tuning
    5. Is cost optimization the primary driver? -> Distillation
    """
    # TODO: Implement the decision logic
    # Consider:
    # - knowledge needs vs behavior needs
    # - knowledge update frequency (static knowledge can be fine-tuned,
    #   frequently changing knowledge needs RAG)
    # - data availability (< 50 examples: prompt engineering or synthetic data,
    #   50-200: synthetic augmentation + fine-tuning,
    #   200+: direct fine-tuning)
    # - budget constraints
    # - latency requirements (fine-tuned models can use shorter prompts = faster)
    raise NotImplementedError("Implement select_approach")


def verify_exercise_1():
    """Verify approach selection logic."""
    # Scenario 1: Simple formatting task with enough data
    s1 = Scenario(
        description="Convert free-text bug reports into structured JSON",
        needs_private_knowledge=False,
        knowledge_update_frequency="static",
        needs_custom_output_format=True,
        needs_custom_tone_or_style=False,
        labeled_examples_available=500,
        monthly_request_volume=10000,
        max_latency_ms=2000,
        budget_usd=1000,
    )
    r1 = select_approach(s1)
    assert r1["approach"] in ("fine_tuning", "prompt_engineering"), \
        f"Scenario 1: Expected fine_tuning or prompt_engineering, got {r1['approach']}"
    assert len(r1["reasoning"]) > 0, "Must provide reasoning"

    # Scenario 2: FAQ bot over company docs that change weekly
    s2 = Scenario(
        description="Answer employee questions about company policies",
        needs_private_knowledge=True,
        knowledge_update_frequency="weekly",
        needs_custom_output_format=False,
        needs_custom_tone_or_style=False,
        labeled_examples_available=0,
        monthly_request_volume=5000,
        max_latency_ms=3000,
        budget_usd=500,
    )
    r2 = select_approach(s2)
    assert r2["approach"] == "rag", \
        f"Scenario 2: Expected rag (needs private, changing knowledge), got {r2['approach']}"

    # Scenario 3: Needs both knowledge and custom behavior
    s3 = Scenario(
        description="Medical chatbot with specific clinical response format",
        needs_private_knowledge=True,
        knowledge_update_frequency="weekly",
        needs_custom_output_format=True,
        needs_custom_tone_or_style=True,
        labeled_examples_available=2000,
        monthly_request_volume=50000,
        max_latency_ms=2000,
        budget_usd=5000,
    )
    r3 = select_approach(s3)
    assert r3["approach"] == "rag_plus_fine_tuning", \
        f"Scenario 3: Expected rag_plus_fine_tuning, got {r3['approach']}"

    # Scenario 4: No data, no special requirements
    s4 = Scenario(
        description="General-purpose Q&A bot",
        needs_private_knowledge=False,
        knowledge_update_frequency="static",
        needs_custom_output_format=False,
        needs_custom_tone_or_style=False,
        labeled_examples_available=0,
        monthly_request_volume=1000,
        max_latency_ms=5000,
        budget_usd=100,
    )
    r4 = select_approach(s4)
    assert r4["approach"] == "prompt_engineering", \
        f"Scenario 4: Expected prompt_engineering, got {r4['approach']}"

    print("Exercise 1 PASSED")


# ============================================================================
# EXERCISE 2: Data Preparation Pipeline
#
# Given raw, messy data, clean it, validate it, format it, and split it
# for training. This is the most time-consuming part of real fine-tuning
# work and the most likely to determine success or failure.
# ============================================================================


def prepare_for_fine_tuning(
    raw_examples: list[dict],
    output_format: str = "openai",
    system_prompt: str = "You are a helpful assistant.",
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> dict[str, list[dict]]:
    """Prepare raw data for fine-tuning.

    Input: list of dicts with "input" and "output" keys (may be messy)
    Output: dict with "train", "validation", "test" splits in the
    requested format

    Steps:
    1. Clean text (normalize whitespace, remove control characters)
    2. Filter out invalid examples (empty input or output)
    3. Deduplicate (by input text -- same input shouldn't appear twice)
    4. Convert to the requested format ("openai" or "alpaca")
    5. Validate format compliance
    6. Split into train/validation/test

    Return the splits dict. Print statistics about each step.
    """
    # TODO: Implement the full pipeline
    #
    # Cleaning hints:
    # - Collapse multiple whitespace characters into one
    # - Strip leading/trailing whitespace
    # - Remove null bytes and zero-width characters
    #
    # Validation hints:
    # - Both input and output must be non-empty after cleaning
    # - Output should be at least 2 words (single-word outputs are suspicious)
    #
    # Deduplication hints:
    # - Deduplicate by normalized input text (lowercase, stripped)
    # - Keep the first occurrence
    #
    # Format hints:
    # - "openai" format: {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
    # - "alpaca" format: {"instruction": system_prompt, "input": ..., "output": ...}
    raise NotImplementedError("Implement prepare_for_fine_tuning")


def verify_exercise_2():
    """Verify data preparation pipeline."""
    raw = [
        {"input": "What is Python?", "output": "Python is a programming language."},
        {"input": "  What  is  Python?  ", "output": "Python is a programming language."},  # Duplicate after cleaning
        {"input": "Explain LoRA", "output": "LoRA is Low-Rank Adaptation, a fine-tuning method."},
        {"input": "", "output": "This has no input"},  # Invalid: empty input
        {"input": "What is ML?", "output": ""},  # Invalid: empty output
        {"input": "What is\x00 RAG?", "output": "RAG is retrieval-augmented generation."},  # Needs cleaning
        {"input": "Define batch size", "output": "OK"},  # Suspicious: single-word output
    ]

    result = prepare_for_fine_tuning(raw, output_format="openai")

    # Should have train, validation, test splits
    assert "train" in result, "Missing 'train' split"
    assert "validation" in result, "Missing 'validation' split"
    assert "test" in result, "Missing 'test' split"

    # Total examples should be less than raw (after filtering and dedup)
    total = sum(len(v) for v in result.values())
    assert total < len(raw), f"Expected fewer examples after cleaning, got {total}"

    # Should be in OpenAI format
    if result["train"]:
        assert "messages" in result["train"][0], "Expected OpenAI chat format"
        messages = result["train"][0]["messages"]
        roles = [m["role"] for m in messages]
        assert "system" in roles, "Missing system message"
        assert "user" in roles, "Missing user message"
        assert "assistant" in roles, "Missing assistant message"

    print("Exercise 2 PASSED")


# ============================================================================
# EXERCISE 3: Synthetic Data Generation Pipeline Design
#
# Design a pipeline that generates training data for a specific domain
# using a teacher model. You won't call an actual API -- instead, build
# the prompts, configuration, and quality filtering logic.
# ============================================================================


@dataclass
class SyntheticPipelineConfig:
    """Configuration for a synthetic data generation pipeline."""

    domain: str
    task_type: str  # "classification", "generation", "extraction"
    num_examples_target: int
    seed_examples: list[dict]  # Real examples to anchor the distribution
    output_labels: list[str] | None = None  # For classification tasks
    quality_threshold: float = 0.7
    teacher_model: str = "gpt-4o"


def design_synthetic_pipeline(
    config: SyntheticPipelineConfig,
) -> dict[str, Any]:
    """Design a synthetic data generation pipeline.

    Return a dict with:
    - "generation_prompts": list of prompt strings for the teacher model
      (at least 3 diverse prompts)
    - "num_batches": how many generation batches to run
    - "examples_per_batch": examples to generate per batch
    - "quality_checks": list of quality check descriptions
    - "estimated_cost_usd": based on teacher model and example count
    - "diversity_strategy": how you ensure diverse outputs
    - "filtering_criteria": what makes an example good vs bad

    The prompts should:
    - Use seed examples to demonstrate the expected format
    - Vary in focus (different difficulty levels, edge cases, etc.)
    - Be specific enough to generate useful data
    """
    # TODO: Implement the pipeline design
    #
    # Key considerations:
    # - Generate 2-3x more examples than you need (filtering will remove some)
    # - Each prompt should have a different "diversity focus"
    #   (common cases, edge cases, adversarial cases, etc.)
    # - Include 2-3 seed examples in each prompt as demonstrations
    # - Quality checks should cover: format compliance, factual plausibility,
    #   diversity (not too similar to other examples), length appropriateness
    # - Cost estimation: ~500 tokens per example, use pricing for teacher model
    raise NotImplementedError("Implement design_synthetic_pipeline")


def verify_exercise_3():
    """Verify synthetic pipeline design."""
    config = SyntheticPipelineConfig(
        domain="customer_support",
        task_type="classification",
        num_examples_target=1000,
        seed_examples=[
            {"input": "My order hasn't arrived yet", "label": "shipping"},
            {"input": "I want a refund", "label": "refund"},
            {"input": "How do I change my password?", "label": "account"},
        ],
        output_labels=["shipping", "refund", "account", "product", "billing"],
        teacher_model="gpt-4o-mini",
    )

    pipeline = design_synthetic_pipeline(config)

    assert "generation_prompts" in pipeline, "Missing generation_prompts"
    assert len(pipeline["generation_prompts"]) >= 3, "Need at least 3 diverse prompts"
    assert pipeline["num_batches"] > 0, "Need at least 1 batch"
    assert pipeline["examples_per_batch"] > 0, "Need examples per batch"

    # Should generate more than target (to account for filtering)
    total_generated = pipeline["num_batches"] * pipeline["examples_per_batch"]
    assert total_generated >= config.num_examples_target, \
        f"Generate at least {config.num_examples_target} examples (got plan for {total_generated})"

    assert "quality_checks" in pipeline, "Missing quality_checks"
    assert len(pipeline["quality_checks"]) >= 3, "Need at least 3 quality checks"
    assert "estimated_cost_usd" in pipeline, "Missing cost estimate"
    assert pipeline["estimated_cost_usd"] > 0, "Cost should be positive"

    # Prompts should reference the seed examples
    all_prompts = " ".join(pipeline["generation_prompts"])
    assert "shipping" in all_prompts.lower() or "order" in all_prompts.lower(), \
        "Prompts should reference seed examples"

    print("Exercise 3 PASSED")


# ============================================================================
# EXERCISE 4: Evaluation Harness
#
# Build a harness that compares a base model (with prompting) against a
# fine-tuned model on the same test set. Compute metrics and determine
# whether the fine-tuning was worth the investment.
# ============================================================================


@dataclass
class ModelOutput:
    """A single model prediction with metadata."""

    input_text: str
    expected_output: str
    predicted_output: str
    latency_ms: float
    token_count: int


def evaluate_and_compare(
    test_set: list[dict[str, str]],
    baseline_outputs: list[ModelOutput],
    finetuned_outputs: list[ModelOutput],
    task_type: str = "classification",
    training_cost_usd: float = 0.0,
) -> dict[str, Any]:
    """Compare fine-tuned model against baseline on a test set.

    Return a dict with:
    - "baseline_metrics": dict of metric_name -> value
    - "finetuned_metrics": dict of metric_name -> value
    - "improvement": dict of metric_name -> {"absolute": ..., "relative_pct": ...}
    - "latency_comparison": {"baseline_avg_ms": ..., "finetuned_avg_ms": ..., "speedup": ...}
    - "cost_analysis": {"training_cost": ..., "inference_savings_per_1k": ..., "break_even_requests": ...}
    - "recommendation": "ship_finetuned" | "keep_baseline" | "needs_more_data"
    - "reasoning": str explaining the recommendation

    Metrics to compute based on task_type:
    - "classification": exact_match, per_class_accuracy
    - "generation": rouge_l (implement a simple version)
    - "extraction": exact_match, partial_match (check if expected is substring of predicted)

    Cost analysis:
    - Calculate per-request cost difference based on token counts
    - Determine break-even point (how many requests to recoup training cost)
    """
    # TODO: Implement the evaluation harness
    #
    # Hints:
    # - Exact match: normalize both strings (lowercase, strip) before comparing
    # - ROUGE-L: longest common subsequence / reference length
    # - Per-class accuracy: group by expected output, compute accuracy per group
    # - Latency comparison: average latency for each model
    # - Cost savings: fine-tuned models often use fewer tokens (shorter prompts needed)
    #   Estimate savings as (baseline_avg_tokens - finetuned_avg_tokens) * cost_per_token
    # - Break-even: training_cost / per_request_savings
    # - Recommendation logic:
    #   - "ship_finetuned" if quality improved by >5% OR latency improved by >20%
    #   - "keep_baseline" if no meaningful improvement
    #   - "needs_more_data" if improvement is marginal (1-5%)
    raise NotImplementedError("Implement evaluate_and_compare")


def verify_exercise_4():
    """Verify evaluation harness."""
    test_set = [
        {"input": "Great product!", "expected": "positive"},
        {"input": "Terrible quality", "expected": "negative"},
        {"input": "It's okay", "expected": "neutral"},
        {"input": "Love it!", "expected": "positive"},
        {"input": "Waste of money", "expected": "negative"},
        {"input": "Average product", "expected": "neutral"},
    ]

    # Baseline: gets 4/6 right, slower, more tokens (longer prompts)
    baseline_outputs = [
        ModelOutput("Great product!", "positive", "positive", 500, 150),
        ModelOutput("Terrible quality", "negative", "negative", 480, 145),
        ModelOutput("It's okay", "neutral", "positive", 520, 155),  # Wrong
        ModelOutput("Love it!", "positive", "positive", 490, 148),
        ModelOutput("Waste of money", "negative", "positive", 510, 152),  # Wrong
        ModelOutput("Average product", "neutral", "neutral", 495, 147),
    ]

    # Fine-tuned: gets 5/6 right, faster, fewer tokens
    finetuned_outputs = [
        ModelOutput("Great product!", "positive", "positive", 200, 80),
        ModelOutput("Terrible quality", "negative", "negative", 190, 75),
        ModelOutput("It's okay", "neutral", "neutral", 210, 82),
        ModelOutput("Love it!", "positive", "positive", 195, 78),
        ModelOutput("Waste of money", "negative", "negative", 205, 80),
        ModelOutput("Average product", "neutral", "positive", 198, 77),  # Wrong
    ]

    result = evaluate_and_compare(
        test_set,
        baseline_outputs,
        finetuned_outputs,
        task_type="classification",
        training_cost_usd=50.0,
    )

    assert "baseline_metrics" in result, "Missing baseline_metrics"
    assert "finetuned_metrics" in result, "Missing finetuned_metrics"
    assert "improvement" in result, "Missing improvement"
    assert "latency_comparison" in result, "Missing latency_comparison"
    assert "recommendation" in result, "Missing recommendation"

    # Fine-tuned should score higher
    baseline_em = result["baseline_metrics"].get("exact_match", 0)
    finetuned_em = result["finetuned_metrics"].get("exact_match", 0)
    assert finetuned_em > baseline_em, \
        f"Fine-tuned ({finetuned_em}) should outperform baseline ({baseline_em})"

    # Fine-tuned should be faster
    assert result["latency_comparison"]["finetuned_avg_ms"] < \
           result["latency_comparison"]["baseline_avg_ms"], \
        "Fine-tuned should have lower latency"

    # Should recommend shipping the fine-tuned model
    assert result["recommendation"] in ("ship_finetuned", "needs_more_data"), \
        f"Expected ship_finetuned or needs_more_data, got {result['recommendation']}"

    print("Exercise 4 PASSED")


# ============================================================================
# EXERCISE 5: Training Cost and GPU Requirements Calculator
#
# Given a model size, dataset, and hardware constraints, calculate the
# training cost, time, and recommend the right infrastructure.
# ============================================================================


def calculate_training_requirements(
    model_size_b: float,
    dataset_size: int,
    avg_example_tokens: int,
    num_epochs: int,
    available_gpus: list[dict],  # [{"name": "rtx_4090", "vram_gb": 24, "count": 1}, ...]
    budget_usd: float,
) -> dict[str, Any]:
    """Calculate training requirements and recommend infrastructure.

    Return a dict with:
    - "recommended_method": "full", "lora", or "qlora"
    - "recommended_gpu": which GPU from available_gpus to use
    - "num_gpus_needed": how many of that GPU
    - "fits_in_budget": bool
    - "estimated_hours": training time estimate
    - "estimated_cost_usd": total cost
    - "memory_breakdown": {"model_gb": ..., "lora_gb": ..., "optimizer_gb": ..., "total_gb": ...}
    - "warnings": list of potential issues

    Memory estimation formulas:
    - Full fine-tune FP16: model (2 bytes/param) + gradients (2 bytes/param) + optimizer (8 bytes/param) = ~12 bytes/param
    - LoRA FP16: model (2 bytes/param) + LoRA params (~0.5-2% of model) + optimizer for LoRA only
    - QLoRA: model (0.5 bytes/param in 4-bit) + LoRA in FP16 + optimizer for LoRA

    Training time estimation:
    - total_tokens = dataset_size * avg_example_tokens * num_epochs
    - tokens_per_second depends on method and GPU (see estimates below)

    GPU throughput (tokens/second, approximate):
    - A100-80GB: full=3000, lora=5000, qlora=4000
    - A100-40GB: full=2000, lora=3500, qlora=3000
    - RTX 4090: full=N/A, lora=2000, qlora=1500
    - RTX 3090: full=N/A, lora=1500, qlora=1000
    - H100: full=6000, lora=10000, qlora=8000
    """
    # TODO: Implement the calculator
    #
    # Steps:
    # 1. For each available GPU, determine which methods are feasible
    #    (does the model fit in VRAM with that method?)
    # 2. Choose the cheapest feasible option that fits the budget
    # 3. Calculate training time and cost
    # 4. Add warnings for potential issues:
    #    - "Dataset may be too small for the number of epochs (risk of overfitting)"
    #      if dataset_size < 500 and num_epochs > 3
    #    - "Full fine-tuning on this model size requires significant compute"
    #      if method is "full" and model_size_b > 13
    #    - "QLoRA may have slightly lower quality than LoRA"
    #      if method is "qlora"
    raise NotImplementedError("Implement calculate_training_requirements")


def verify_exercise_5():
    """Verify training requirements calculator."""
    gpus = [
        {"name": "rtx_4090", "vram_gb": 24, "count": 2, "cost_per_hour": 0.50},
        {"name": "a100_80gb", "vram_gb": 80, "count": 1, "cost_per_hour": 2.50},
    ]

    # Small model, should fit on RTX 4090 with QLoRA
    result = calculate_training_requirements(
        model_size_b=7,
        dataset_size=1000,
        avg_example_tokens=512,
        num_epochs=3,
        available_gpus=gpus,
        budget_usd=100,
    )

    assert "recommended_method" in result, "Missing recommended_method"
    assert result["recommended_method"] in ("lora", "qlora"), \
        f"7B model should use LoRA or QLoRA, got {result['recommended_method']}"
    assert "estimated_hours" in result, "Missing estimated_hours"
    assert result["estimated_hours"] > 0, "Training time should be positive"
    assert "estimated_cost_usd" in result, "Missing estimated_cost"
    assert "memory_breakdown" in result, "Missing memory_breakdown"

    # Large model, needs bigger GPU
    result_large = calculate_training_requirements(
        model_size_b=70,
        dataset_size=5000,
        avg_example_tokens=1024,
        num_epochs=1,
        available_gpus=gpus,
        budget_usd=500,
    )

    assert result_large["recommended_gpu"]["name"] == "a100_80gb", \
        "70B model should use A100"
    assert result_large["recommended_method"] == "qlora", \
        "70B model on single A100 should use QLoRA"

    print("Exercise 5 PASSED")


# ============================================================================
# EXERCISE 6: LoRA Configuration Designer
#
# Given constraints (GPU RAM, quality target, data size, task complexity),
# design an optimal LoRA configuration with justification for each choice.
# ============================================================================


def design_lora_config(
    model_size_b: float,
    gpu_vram_gb: int,
    dataset_size: int,
    task_complexity: str,  # "simple", "moderate", "complex"
    quality_target: str,  # "good_enough", "high", "maximum"
    inference_latency_matters: bool = False,
) -> dict[str, Any]:
    """Design a LoRA configuration based on constraints.

    Return a dict with:
    - "rank": int (4, 8, 16, 32, 64, or 128)
    - "alpha": int (typically 2x rank)
    - "target_modules": list of module names
    - "dropout": float
    - "use_4bit": bool (QLoRA)
    - "justification": dict mapping each parameter to a string explaining why
    - "estimated_trainable_params_pct": float
    - "estimated_vram_gb": float

    Design principles:
    - Rank: higher for complex tasks, lower for simple ones. But constrained
      by data size (high rank + small data = overfitting).
    - Target modules: more modules = more capacity but more VRAM.
      Minimum: q_proj, v_proj. Standard: q,k,v,o projections.
      Maximum: all attention + MLP layers.
    - Dropout: higher for small datasets (regularization), lower for large.
    - 4-bit: use if VRAM is tight. Avoid if quality_target is "maximum".
    - Alpha: standard is 2x rank. Higher alpha = larger effective learning rate
      for the adapter.
    """
    # TODO: Implement the configuration designer
    #
    # Decision logic:
    #
    # Rank selection:
    #   simple task + small data → 8
    #   simple task + large data → 16
    #   moderate task → 16 or 32
    #   complex task + enough data → 32 or 64
    #   complex task + small data → 16 (prevent overfitting)
    #   "maximum" quality → double the above
    #
    # Target modules:
    #   "good_enough" quality → ["q_proj", "v_proj"]
    #   "high" quality → ["q_proj", "k_proj", "v_proj", "o_proj"]
    #   "maximum" quality → ["q_proj", "k_proj", "v_proj", "o_proj",
    #                         "gate_proj", "up_proj", "down_proj"]
    #
    # Dropout:
    #   dataset_size < 500 → 0.1
    #   dataset_size < 5000 → 0.05
    #   dataset_size >= 5000 → 0.0
    #
    # QLoRA decision:
    #   Model FP16 size (model_size_b * 2) > gpu_vram_gb * 0.7 → use 4-bit
    #   quality_target == "maximum" → prefer not using 4-bit
    #
    # VRAM estimation:
    #   QLoRA: model_size_b * 0.5 + lora_overhead (typically 1-3 GB)
    #   LoRA: model_size_b * 2 + lora_overhead
    raise NotImplementedError("Implement design_lora_config")


def verify_exercise_6():
    """Verify LoRA configuration designer."""
    # Simple task, small GPU, moderate data
    config1 = design_lora_config(
        model_size_b=7,
        gpu_vram_gb=24,
        dataset_size=500,
        task_complexity="simple",
        quality_target="good_enough",
    )

    assert "rank" in config1, "Missing rank"
    assert "alpha" in config1, "Missing alpha"
    assert "target_modules" in config1, "Missing target_modules"
    assert "dropout" in config1, "Missing dropout"
    assert "justification" in config1, "Missing justification"
    assert config1["rank"] <= 16, f"Simple task shouldn't need rank > 16, got {config1['rank']}"
    assert config1["dropout"] >= 0.05, f"Small dataset needs dropout, got {config1['dropout']}"
    assert config1["use_4bit"] is True, "7B on 24GB should use QLoRA"

    # Complex task, large GPU, lots of data, maximum quality
    config2 = design_lora_config(
        model_size_b=7,
        gpu_vram_gb=80,
        dataset_size=10000,
        task_complexity="complex",
        quality_target="maximum",
    )

    assert config2["rank"] >= 32, f"Complex task + max quality needs rank >= 32, got {config2['rank']}"
    assert config2["use_4bit"] is False, "Maximum quality on 80GB should avoid QLoRA"
    assert len(config2["target_modules"]) >= 4, \
        f"Maximum quality should target many modules, got {len(config2['target_modules'])}"
    assert config2["dropout"] == 0.0 or config2["dropout"] <= 0.01, \
        f"Large dataset needs minimal dropout, got {config2['dropout']}"

    # Check justifications exist for key parameters
    assert "rank" in config2["justification"], "Missing justification for rank"
    assert "target_modules" in config2["justification"], "Missing justification for target_modules"

    print("Exercise 6 PASSED")


# ============================================================================
# RUN ALL EXERCISES
# ============================================================================


def main():
    exercises = [
        ("Exercise 1: Approach Selection", verify_exercise_1),
        ("Exercise 2: Data Preparation", verify_exercise_2),
        ("Exercise 3: Synthetic Data Pipeline", verify_exercise_3),
        ("Exercise 4: Evaluation Harness", verify_exercise_4),
        ("Exercise 5: Training Cost Calculator", verify_exercise_5),
        ("Exercise 6: LoRA Configuration", verify_exercise_6),
    ]

    passed = 0
    failed = 0

    for name, verify_fn in exercises:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        try:
            verify_fn()
            passed += 1
        except NotImplementedError:
            print(f"  NOT IMPLEMENTED (TODO)")
            failed += 1
        except AssertionError as e:
            print(f"  FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"  Results: {passed}/{passed + failed} exercises passed")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

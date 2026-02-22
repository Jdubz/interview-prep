# Key Terminology

Quick-reference glossary of terms likely to come up in an LLM interview.

---

## Model & Architecture

| Term | Definition |
|---|---|
| **Transformer** | Neural network architecture using self-attention to process sequences in parallel. Foundation of all modern LLMs. |
| **Self-Attention** | Mechanism where each token computes relevance scores against every other token. Uses Query, Key, Value matrices. |
| **Multi-Head Attention** | Running multiple attention computations in parallel, each capturing different relationship types. |
| **Feed-Forward Network (FFN)** | Dense layers in each transformer block that process attention outputs. Where much of the "knowledge" is stored. |
| **Parameters** | The learned weights of the model. "7B parameters" = 7 billion numbers defining the model's behavior. |
| **Decoder-Only** | Architecture that generates text left-to-right (GPT, Llama, Claude). Most modern LLMs are decoder-only. |
| **Encoder-Decoder** | Architecture with separate understanding (encoder) and generation (decoder) stages. Used in T5, original transformers. |

## Tokenization & Input

| Term | Definition |
|---|---|
| **Token** | Subword unit the model processes. Not quite a word, not quite a character — learned from training data. |
| **BPE (Byte Pair Encoding)** | Algorithm for building a token vocabulary by iteratively merging frequent character pairs. |
| **Context Window** | Maximum tokens (input + output) the model handles in one request. Ranges from 4K to 2M+. |
| **Prompt** | The full input to the model: system message + conversation history + user message. |
| **System Prompt** | Instructions that set the model's behavior, role, and constraints. Processed with elevated priority. |

## Generation & Sampling

| Term | Definition |
|---|---|
| **Temperature** | Controls randomness. 0 = deterministic, 1+ = more random. Scales logits before softmax. |
| **Top-p (Nucleus Sampling)** | Only sample from tokens whose cumulative probability ≥ p. Adaptive — narrows when model is confident. |
| **Top-k** | Only consider the k most probable next tokens. |
| **Logits** | Raw model output scores before conversion to probabilities. |
| **Softmax** | Function that converts logits into a probability distribution (sums to 1). |
| **Greedy Decoding** | Always pick the highest-probability token. Equivalent to temperature=0. |
| **Beam Search** | Explore multiple candidate sequences in parallel, pick the best overall. Less common in LLM APIs. |
| **Stop Sequence** | Token or string that signals the model to stop generating. |

## Training

| Term | Definition |
|---|---|
| **Pre-training** | Initial training on massive text data to predict next tokens. Learns language, facts, reasoning. |
| **Fine-Tuning** | Further training on task-specific data to adapt the model. |
| **SFT (Supervised Fine-Tuning)** | Training on (prompt, ideal response) pairs to teach instruction-following. |
| **RLHF** | Reinforcement Learning from Human Feedback. Humans rank outputs → train reward model → optimize LLM. |
| **RLAIF** | Same as RLHF but using AI feedback instead of human raters. |
| **LoRA** | Low-Rank Adaptation. Efficient fine-tuning that trains small adapter matrices instead of all parameters. |
| **Distillation** | Training a small model to mimic a large model's behavior. Cheaper to run, similar quality. |
| **Alignment** | Making models helpful, harmless, and honest through RLHF/RLAIF and other techniques. |

## Embeddings & Retrieval

| Term | Definition |
|---|---|
| **Embedding** | Dense vector representation of text where semantic similarity → geometric proximity. |
| **Cosine Similarity** | Measure of angle between two vectors. 1 = identical direction, 0 = orthogonal. Standard for text similarity. |
| **Vector Database** | Specialized database for storing and querying high-dimensional vectors. Examples: Pinecone, Qdrant, pgvector. |
| **ANN (Approximate Nearest Neighbor)** | Algorithms (HNSW, IVF) that trade perfect accuracy for fast similarity search at scale. |
| **HNSW** | Hierarchical Navigable Small Worlds. Graph-based ANN algorithm. Most popular in vector DBs. |
| **Semantic Search** | Finding documents by meaning rather than keyword matching. Uses embeddings. |
| **BM25** | Classic keyword-based ranking algorithm. Complements vector search in hybrid systems. |

## RAG

| Term | Definition |
|---|---|
| **RAG (Retrieval-Augmented Generation)** | Augmenting LLM responses with retrieved documents. Reduces hallucination, enables domain-specific knowledge. |
| **Chunking** | Splitting documents into smaller pieces for embedding and retrieval. |
| **Reranking** | Using a cross-encoder to re-score initial retrieval results for better precision. |
| **Hybrid Search** | Combining vector (semantic) and keyword (BM25) search for better recall. |
| **HyDE** | Hypothetical Document Embeddings. Generate a fake answer, embed it, use for retrieval. |
| **Cross-Encoder** | Model that processes (query, document) pairs jointly for accurate relevance scoring. Slower but more accurate than bi-encoders. |

## Agents & Tool Use

| Term | Definition |
|---|---|
| **Function Calling / Tool Use** | LLM outputs structured requests for external actions instead of just text. |
| **Agent** | LLM in a loop: reason → call tools → process results → repeat until done. |
| **Agent Loop** | The cycle of LLM call → tool execution → result injection → next LLM call. |
| **ReAct** | Reasoning + Acting pattern. Model alternates between Thought, Action, and Observation steps. |
| **Tool Schema** | JSON definition of a tool (name, description, parameters) that tells the model what's available. |
| **Guardrails** | Constraints on agent behavior: input validation, permission checks, output filtering. |
| **Human-in-the-Loop** | Requiring human approval for high-stakes agent actions before execution. |

## Production

| Term | Definition |
|---|---|
| **Streaming** | Returning tokens as they're generated via SSE. Reduces perceived latency. |
| **TTFT (Time to First Token)** | Latency until the first token appears. Key UX metric for streaming. |
| **Prompt Caching** | Provider-side caching of repeated prompt prefixes for reduced cost/latency. |
| **Structured Output** | Constraining model output to match a schema (JSON, XML). Provider-enforced or prompt-based. |
| **Hallucination** | Model generating plausible but incorrect information. Fundamental LLM limitation. |
| **Prompt Injection** | Adversarial input that manipulates the model's behavior by overriding instructions. |
| **Eval** | Systematic evaluation of LLM output quality using test sets and metrics. |
| **LLM-as-Judge** | Using a powerful LLM to evaluate another LLM's outputs. Scalable quality assessment. |
| **Observability** | Logging, tracing, and monitoring LLM requests for debugging and optimization. |

## Emerging / Advanced

| Term | Definition |
|---|---|
| **MoE (Mixture of Experts)** | Architecture where only a subset of parameters activate per token. Enables larger models at similar compute cost. |
| **Multimodal** | Models that process multiple input types (text, images, audio, video). |
| **Context Distillation** | Compressing long context into shorter representations while preserving information. |
| **Constitutional AI** | Training models with a set of principles (a "constitution") to self-critique and self-improve. |
| **Agentic AI** | Systems where LLMs autonomously plan, execute, and adapt to complete complex tasks. |
| **MCP (Model Context Protocol)** | Standardized protocol for connecting LLMs to external tools and data sources. |

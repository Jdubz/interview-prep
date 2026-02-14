# LLM + Python Interview Prep

**Learn Python and LLM concepts together.** This study guide prepares you for forward deployed engineer, LLM engineer, and AI product roles—while teaching you modern Python from scratch.

> **For TypeScript developers**: This guide is designed for senior engineers with strong fundamentals in other languages. Python concepts are explained with comparisons to TypeScript where helpful.

**What makes this different:** Most LLM guides assume Python expertise. This one teaches both simultaneously—Python fundamentals grow alongside LLM complexity.

---

## Who This Is For

- **Senior engineers** from other languages (TypeScript, Java, C#) preparing for Python-based LLM roles
- **Python developers** who want to add LLM/AI skills
- **Anyone** preparing for interviews at companies building gen AI products

**You'll learn:**
- Python from basics (variables, functions) to advanced (async, protocols, generics)
- LLM foundations (transformers, tokenization, context windows)
- Practical skills (prompt engineering, RAG, agents, production deployment)
- Interview prep (questions + answers for both Python and LLMs)

---

## Learning Path

### Prerequisites
✅ Programming fundamentals (any language)
✅ Understanding of APIs and web services
✅ Basic command line skills

### What You'll Learn

**Python Track** (grows progressively):
- **Basics**: Variables, types, functions, classes, control flow
- **Modern Python**: Type hints, dataclasses, protocols, async/await
- **Intermediate**: Generators, comprehensions, decorators, JSON handling
- **Advanced**: Async orchestration, error handling, generics, state management
- **Production**: Logging, testing, monitoring, type checking, deployment

**LLM Track** (integrated with Python):
- **Foundations**: How LLMs work, tokenization, parameters
- **Prompt Engineering**: Techniques, patterns, structured output
- **RAG**: Embeddings, vector search, chunking, retrieval
- **Agents**: Tool use, ReAct pattern, orchestration
- **Production**: Streaming, caching, evals, safety, observability

---

## Study Plans

### 🚀 Fast Track (8-12 hours)

**Focus on fundamentals and interview prep:**

| Order | File | Time | Focus |
|---|---|---|---|
| 1 | `00-python-quickstart/basics.md` | 60 min | Python fundamentals |
| 2 | `00-python-quickstart/modern-python.md` | 45 min | Type hints, dataclasses, async |
| 3 | `01-foundations/concepts.md` | 45 min | How LLMs work |
| 4 | `02-prompt-engineering/techniques.md` | 45 min | Core prompt patterns |
| 5 | `03-rag-and-embeddings/concepts.md` | 40 min | RAG architecture |
| 6 | `04-agents-and-tool-use/concepts.md` | 30 min | Agent patterns |
| 7 | `06-interview-prep/python-questions.md` | 60 min | Python interview Q&A |
| 8 | `06-interview-prep/llm-questions.md` | 60 min | LLM interview Q&A |
| 9 | `06-interview-prep/system-design.md` | 30 min | System design framework |

**Total: ~8 hours** of focused study covering both Python and LLMs.

---

### 📚 Complete Track (20-25 hours)

**Deep dive into everything:**

#### Week 1: Python + LLM Foundations (8-10 hours)

| # | File | Time | What You'll Learn |
|---|---|---|---|
| 1 | `00-python-quickstart/basics.md` | 60 min | Variables, types, functions, classes, collections |
| 2 | `00-python-quickstart/modern-python.md` | 60 min | Type hints, dataclasses, protocols, async/await |
| 3 | `00-python-quickstart/exercises.py` | 60 min | Practice Python fundamentals |
| 4 | `01-foundations/concepts.md` | 45 min | Transformers, attention, tokenization |
| 5 | `01-foundations/python-for-llms.md` | 45 min | JSON, strings, dicts for LLM work |
| 6 | `01-foundations/exercises.py` | 45 min | Practice with LLM data structures |
| 7 | `02-prompt-engineering/techniques.md` | 45 min | Zero-shot, few-shot, CoT, structured output |
| 8 | `02-prompt-engineering/patterns.md` | 30 min | Classification, extraction, summarization |
| 9 | `02-prompt-engineering/python-essentials.md` | 45 min | F-strings, templates, string manipulation |
| 10 | `02-prompt-engineering/examples.py` | 45 min | Read through prompt engineering code |

#### Week 2: RAG + Agents (8-10 hours)

| # | File | Time | What You'll Learn |
|---|---|---|---|
| 11 | `03-rag-and-embeddings/concepts.md` | 40 min | Embeddings, similarity, chunking |
| 12 | `03-rag-and-embeddings/architecture.md` | 40 min | RAG pipeline, vector DBs, hybrid search |
| 13 | `03-rag-and-embeddings/python-intermediate.md` | 60 min | Generators, async, vector operations |
| 14 | `03-rag-and-embeddings/examples.py` | 40 min | RAG pipeline implementation |
| 15 | `04-agents-and-tool-use/concepts.md` | 30 min | Function calling, tool schemas |
| 16 | `04-agents-and-tool-use/patterns.md` | 30 min | ReAct, orchestration, guardrails |
| 17 | `04-agents-and-tool-use/python-advanced.md` | 60 min | Advanced async, error handling, generics |
| 18 | `04-agents-and-tool-use/examples.py` | 45 min | Agent implementation |

#### Week 3: Production + Interview Prep (4-5 hours)

| # | File | Time | What You'll Learn |
|---|---|---|---|
| 19 | `05-production-concerns/guide.md` | 40 min | Streaming, caching, evals, safety |
| 20 | `05-production-concerns/python-production.md` | 60 min | Logging, testing, monitoring, deployment |
| 21 | `05-production-concerns/examples.py` | 30 min | Production patterns |
| 22 | `06-interview-prep/python-questions.md` | 60 min | Python interview questions & answers |
| 23 | `06-interview-prep/llm-questions.md` | 45 min | LLM interview questions & answers |
| 24 | `06-interview-prep/system-design.md` | 30 min | LLM system design framework |

**Total: ~22 hours** covering Python from basics to production + comprehensive LLM knowledge.

---

## File Structure

```
llm-prep/
├── README.md                               ← You are here
│
├── 00-python-quickstart/                   ← Start here if new to Python
│   ├── basics.md                           # Variables, types, functions, classes
│   ├── modern-python.md                    # Type hints, dataclasses, async
│   └── exercises.py                        # Practice problems
│
├── 01-foundations/                         ← LLM basics + Python for LLMs
│   ├── concepts.md                         # How LLMs work
│   ├── key-terminology.md                  # Quick reference glossary
│   ├── python-for-llms.md                  # JSON, strings, dicts for LLM work
│   └── exercises.py                        # Practice with API responses
│
├── 02-prompt-engineering/                  ← Prompt engineering + string manipulation
│   ├── techniques.md                       # Core techniques (few-shot, CoT, etc.)
│   ├── patterns.md                         # Applied patterns
│   ├── python-essentials.md                # F-strings, templates, formatting
│   └── examples.py                         # Annotated Python code
│
├── 03-rag-and-embeddings/                  ← RAG + intermediate Python
│   ├── concepts.md                         # Embeddings, chunking, similarity
│   ├── architecture.md                     # RAG pipeline architecture
│   ├── python-intermediate.md              # Generators, async, vector ops
│   └── examples.py                         # RAG pipeline implementation
│
├── 04-agents-and-tool-use/                 ← Agents + advanced Python
│   ├── concepts.md                         # Tool use, function calling
│   ├── patterns.md                         # ReAct, orchestration
│   ├── python-advanced.md                  # Advanced async, error handling
│   └── examples.py                         # Agent implementation
│
├── 05-production-concerns/                 ← Production + Python best practices
│   ├── guide.md                            # Production patterns for LLMs
│   ├── python-production.md                # Logging, testing, monitoring
│   └── examples.py                         # Production code patterns
│
└── 06-interview-prep/                      ← Interview preparation
    ├── python-questions.md                 # Python interview Q&A
    ├── llm-questions.md                    # LLM interview Q&A
    └── system-design.md                    # System design framework
```

---

## Python Concepts Covered

Studying these materials is also a structured way to learn modern Python:

| Python Feature | Where It's Taught | Complexity |
|---|---|---|
| **Basics** | | |
| Variables, types, functions | `00-python-quickstart/basics.md` | Beginner |
| Lists, dicts, tuples, sets | `00-python-quickstart/basics.md` | Beginner |
| Control flow (if/for/while) | `00-python-quickstart/basics.md` | Beginner |
| List comprehensions | `00-python-quickstart/basics.md` | Beginner |
| **Modern Python** | | |
| Type hints (3.10+ syntax) | `00-python-quickstart/modern-python.md` | Intermediate |
| Dataclasses | `00-python-quickstart/modern-python.md` | Intermediate |
| Protocols (structural typing) | `00-python-quickstart/modern-python.md` | Intermediate |
| Async/await basics | `00-python-quickstart/modern-python.md` | Intermediate |
| Pattern matching | `00-python-quickstart/modern-python.md` | Intermediate |
| **Intermediate** | | |
| Generators & yield | `03-rag-and-embeddings/python-intermediate.md` | Intermediate |
| Generator expressions | `03-rag-and-embeddings/python-intermediate.md` | Intermediate |
| Async iterators | `03-rag-and-embeddings/python-intermediate.md` | Intermediate |
| asyncio.gather() | `03-rag-and-embeddings/python-intermediate.md` | Intermediate |
| Context managers | `03-rag-and-embeddings/python-intermediate.md` | Intermediate |
| **Advanced** | | |
| Advanced async patterns | `04-agents-and-tool-use/python-advanced.md` | Advanced |
| Generics (TypeVar) | `04-agents-and-tool-use/python-advanced.md` | Advanced |
| Decorators | `04-agents-and-tool-use/python-advanced.md` | Advanced |
| Error handling strategies | `04-agents-and-tool-use/python-advanced.md` | Advanced |
| State management | `04-agents-and-tool-use/python-advanced.md` | Advanced |
| **Production** | | |
| Structured logging | `05-production-concerns/python-production.md` | Advanced |
| Testing (pytest, mocks) | `05-production-concerns/python-production.md` | Advanced |
| Type checking (mypy) | `05-production-concerns/python-production.md` | Advanced |
| Performance profiling | `05-production-concerns/python-production.md` | Advanced |
| Environment config | `05-production-concerns/python-production.md` | Advanced |

---

## How to Use This Guide

### Reading the Materials

1. **Markdown first, code second**: Read concept files (`.md`) before code files (`.py`)
2. **Practice out loud**: Explain concepts as if teaching someone
3. **Type hints are your friend**: The code uses modern Python 3.10+ syntax
4. **Code is meant to be read, not run**: Examples show patterns, not complete applications

### Practicing

1. **Do the exercises**: Each section has practice problems (`.py` files)
2. **Code along**: Don't just read—type out examples
3. **Modify and experiment**: Change parameters, try edge cases
4. **Build something**: Apply concepts to a small project

### Interview Prep

1. **Practice explaining**: Use the Q&A sections to practice verbal explanations
2. **Time yourself**: Can you explain key concepts in 2-3 minutes?
3. **Use the system design framework**: Practice designing LLM systems
4. **Code on a whiteboard**: Practice without autocomplete

---

## Tips for Success

### General Study Tips

- **Interleave topics**: Mix Python and LLM concepts, don't do all Python then all LLMs
- **Space repetition**: Review concepts multiple times over days
- **Teach back**: Explain concepts to a friend (or rubber duck)
- **Build projects**: Best way to solidify knowledge

### For TypeScript Developers

- **Embrace the differences**: Python is not JavaScript with different syntax
- **snake_case, not camelCase**: Get used to Python naming conventions
- **Indentation matters**: No braces—4 spaces define blocks
- **Dynamic typing**: Type hints are optional (but recommended)
- **Multiple return values**: Uses tuples, not objects

### For Interviews

- **Start simple**: Don't jump to complex solutions
- **Think aloud**: Explain your reasoning as you code
- **Ask questions**: Clarify requirements before coding
- **Consider edge cases**: Empty inputs, None values, errors
- **Know your trade-offs**: Time vs space, accuracy vs latency

---

## Customizing Your Path

### If you already know Python:

Skip `00-python-quickstart/` and just review the Python sections for LLM-specific patterns:
- `01-foundations/python-for-llms.md` (JSON handling)
- `02-prompt-engineering/python-essentials.md` (string templates)
- `03-rag-and-embeddings/python-intermediate.md` (async, generators)
- `04-agents-and-tool-use/python-advanced.md` (error handling, generics)

### If you already know LLMs:

Focus on the Python track:
1. `00-python-quickstart/` → Python basics
2. Review all `python-*.md` files in each section
3. Complete `06-interview-prep/python-questions.md`

### If you're short on time:

Do the **Fast Track** (8-12 hours) focusing on concepts and interview prep, skip the code deep-dives.

---

## Making Examples Runnable

The code examples use generic interfaces (Protocols) so they work with any LLM provider. To make them runnable:

```python
# Install your preferred SDK
pip install anthropic  # or openai, cohere, etc.

# Implement the CompletionFn protocol
from anthropic import Anthropic

class AnthropicCompletion:
    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)

    async def __call__(self, messages: list[Message], config: LLMConfig) -> str:
        response = self.client.messages.create(
            model=config.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            max_tokens=config.max_tokens or 1024,
            temperature=config.temperature
        )
        return response.content[0].text

# Now examples work!
complete = AnthropicCompletion(api_key="...")
result = await zero_shot_classify(complete, "ticket text")
```

---

## Additional Resources

### Official Documentation
- [Python Documentation](https://docs.python.org/3/)
- [Anthropic API Docs](https://docs.anthropic.com/)
- [OpenAI API Docs](https://platform.openai.com/docs/)

### Python Learning
- [Real Python](https://realpython.com/) - Excellent Python tutorials
- [Python Type Checking Guide](https://realpython.com/python-type-checking/)
- [AsyncIO Documentation](https://docs.python.org/3/library/asyncio.html)

### LLM Learning
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [LangChain Docs](https://python.langchain.com/) - Good for patterns, even if you don't use the library
- [Anthropic Prompt Engineering](https://docs.anthropic.com/en/docs/prompt-engineering)

---

## Contributing

Found an error? Have a suggestion? This is a personal study guide, but feedback is welcome!

---

## License

MIT

---

**Ready to start?** Begin with [`00-python-quickstart/basics.md`](00-python-quickstart/basics.md) or jump to the [Fast Track](#-fast-track-8-12-hours) if you're in a hurry.

**Good luck with your interview prep!** 🚀

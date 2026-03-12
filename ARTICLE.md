# AI with Memory: How a Small Model Works with a Large Project

[🇷🇺 Русская версия](ARTICLE.ru.md)

*Why an 8-billion-parameter model can't hold an entire project in its head — and how a pipeline with context distillation works around it*

---

## The Problem Everyone Ignores

When you open a new chat with an AI assistant, it knows nothing about you. You explain the context again, paste the code again, remind it how the project is structured again. It's frustrating — but most people accept it as a given.

There's an exact analogy for this problem. Imagine a colleague who comes in every morning with a completely wiped memory. A brilliant specialist who picks things up instantly — but doesn't remember yesterday's conversation. Every day you spend the first hour bringing them up to speed all over again.

Cloud models (GPT-4, Claude) solve this with brute force — 128K token context, just paste everything in. But if you work with a **local** model on your own GPU — context is limited. Qwen3-8B sees ~8000 tokens at a time. That's 300-400 lines of code. One file — and there's no room left for the response.

---

## The Naive Approach: Stuff Everything into Context

Say the model needs to fix a function. To do this well, it needs to know:
- the code file itself (~3000 tokens)
- style conventions (~800 tokens)
- architectural constraints (~600 tokens)
- project context (~500 tokens)
- the task (~50 tokens)

Total: **~4950 tokens** on input. Out of 8192, about ~3200 remain for the response — barely enough for a diff, and the model starts truncating or losing its train of thought.

You could drop the conventions — but then the model will violate the style. You could drop the architecture — but then it'll break dependencies. A classic trade-off: **more knowledge on input — less room for the response**.

---

## The Idea: Let Each "Expert" Read Its Part and Report the Essentials

Instead of stuffing everything into one request, we make several separate passes. Each agent receives a file + its own knowledge and **compresses** it into a short memo — 2-4 sentences.

The coder at the end receives the file + focused memos instead of all the source documents:

```
Task: "fix the bayes_update function"
  │
  ├─ [1] Navigator        → "bayes_update() lines 42-67, depends on validate_input()"
  │
  ├─ [2] analyst          → "replace X with Y at line 45, add validation at Z"
  │       (analyzes code + task → specific plan)
  │
  ├─ [3] context agent    → "project goal — demo of Bayesian methods"
  │       (knowledge from agents/context/)
  │
  ├─ [4] arch agent       → "don't break the float→float signature, dependencies X→Y"
  │       (knowledge from agents/arch/)
  │
  ├─ [5] style agent      → "snake_case, type hints required, docstring"
  │       (knowledge from agents/style/)
  │
  ├─ [6] coder            → unified diff following the analyst's plan + agent constraints
  │
  └─ [7] reviewer         → validates diff against code structure
            │
            ▼
         /patch → file updated
```

**The Math:**

Each agent works in an isolated eval pass and sees the full context of its knowledge. But the output is a memo of ~80 tokens. Analyst + three agents → 320 tokens instead of 1900. The coder receives:

```
file 3000 + 4 memos 320 + task 50 = 3370 tokens → ~4800 for the response
```

Instead of 3200 — **4800 tokens** for code generation. The model gets the same knowledge, but compressed — and has headroom for a full diff. The key point: the analyst gives the coder a **specific plan** of what to change, not just constraints.

This is not RAG (retrieval-augmented generation) — the model doesn't search a vector database. This is **context distillation**: each agent processes its knowledge into a focused hint for the coder.

---

## How the Model Finds the Right Files

When a user types "fix the bayes_update function," the system needs to figure out which file contains that function and what other files are nearby. For this, it uses tree-sitter — a parser that builds a syntax tree for every Python file in the project.

**Symbols** are extracted from the tree: function names, classes, methods — with line numbers and signatures. A BM25 index (classic text search) is built on top of them. The user's query is ranked by symbol relevance, and the system finds the right files automatically.

This is not semantic search — if a function is called `process_data` and the task is described as "fix data processing," BM25 might miss. But for specific names it works precisely and fast, with no GPU cost for embeddings.

---

## Agents — A Knowledge Base in Git

Where do agents get their knowledge? From markdown files in the client project's git repository:

```
agents/
  style/global.md      ← naming, formatting, restrictions
  arch/overview.md     ← modules, dependencies, data flow
  context/project.md   ← project goal, business rules
```

These are plain text files written by the team. The style agent knows your conventions, the arch agent knows your modules and dependencies, the context agent knows why the project exists in the first place.

Agents are exactly as useful as the knowledge described in them. Empty template → empty memo. Detailed description → precise hint for the coder.

The pipeline is configured via `pipeline.json` — you can change the order, disable stages, add your own agents. Need a security agent? Create `agents/security/global.md` and add a stage.

---

## KV-Cache: Acceleration, Not Expansion

Besides the pipeline, there's also a manual mode — load a specific file and ask questions directly. A different optimization works here: saving the model's KV-cache to disk.

When the model "reads" a file, it forms an internal state — key/value matrices for each layer. This state can be saved to a file and loaded later in ~1 second instead of re-reading (~15 seconds).

This **does not expand** the context — the file still has to fit within 8192 tokens. But it saves time on repeated access to the same file.

Patterns are tied to the git branch. Switch branches — and the patterns for that branch are loaded. In feature branches, the conversation accumulates between sessions (the model "remembers" previous questions). In main — patterns are recreated on each commit.

---

## What This Doesn't Do

**Doesn't expand context.** The pipeline works around the limitation through distillation, but each individual stage still operates within the 8192-token window. Very large files get truncated.

**No semantic search.** Navigation is text-based (BM25). If the function name doesn't match the task description, the system can miss.

**Python only.** The tree-sitter navigator parses `.py` files. Other languages require adding the corresponding grammars.

**One model for everything.** Navigation, agents, coder, reviewer — all on a single Qwen3-8B. Diff quality is limited by the capabilities of an 8B model.

**Patterns are not portable.** The KV-cache is tied to a specific model and quantization. A different machine with a different version won't load someone else's patterns.

---

## Why This If Claude and GPT-4 Exist?

Fair question. Cloud models are more powerful, context is larger, quality is higher. But:

**Data never leaves the machine.** For companies with NDAs or sensitive code, this isn't a matter of convenience — it's a requirement. A local model on your own GPU is the only option.

**No API keys or subscriptions.** Download the model once (~5 GB) — work as much as you want. No dependency on a third-party service, no limits, no surprise bills.

**Educational value.** The pipeline clearly demonstrates how to work around the limitations of a small model — through task decomposition and context distillation. This is a pattern applicable beyond this specific tool.

---

## How This Applies in Practice

**Fix a function.** Write a task → tree-sitter finds the file → agents add context → coder writes a diff → reviewer checks → `/patch`. The entire cycle — one command.

**Analyze a specific file.** Load a file → ask the model questions directly. The conversation accumulates in the KV-cache — the model remembers previous answers.

**Review against conventions.** `/review @file` runs the file through the style agent. The model checks the code against your rules, not general principles.

**Team knowledge base.** `agents/` with architecture, style, and context descriptions is stored in git. A new developer gets the same agents as the rest of the team.

---

## Why This Architecture Looks Like the Brain

During the design of Cognit, there was no conscious intent to replicate neuroscience models. The architecture emerged from a practical question: how to make a small model work effectively with a large codebase. But in retrospect, the result maps almost exactly onto **predictive coding** — a theory of how the brain itself processes information.

**KV-cache as a predictive model.** The brain doesn't store raw sensory data — it maintains a compressed representation of the world that allows it to predict future inputs. A Cognit pattern is the same thing: not a copy of files, but a model that "understands" code and can answer questions about it.

**Stale detection as prediction error.** In predictive coding, the brain notices when its predictions diverge from incoming signals. The post-commit hook compares file hashes against stored representations — a literal mismatch detector between the model's "beliefs" and reality.

**Lazy retrain as Bayesian updating.** The brain doesn't recompute all beliefs when a single stimulus arrives. It updates only those that become relevant. Cognit does the same — the hook marks patterns as stale, but retraining happens only on the next `/ask`, when that knowledge is actually needed.

**The pipeline as hierarchical inference.** Each agent (navigator → analyst → arch → style → coder → reviewer) is a layer that compresses and refines the representation top-down — much like layers of cortex, from sensory input to abstract reasoning and back.

**Agents as specialized cortical regions.** Each agent has its own "world model" (KV-cache) trained on its domain. They don't know everything — they contribute their specific expertise to a shared inference process.

Karl Friston would call this **active inference** — a system that minimizes the gap between its model and reality through two mechanisms: updating the model (retrain) or changing the world (patch). Cognit does both.

This wasn't designed top-down from theory. It was built bottom-up from intuition about how persistent understanding should work. The fact that it converges with Bayesian brain theory suggests that context distillation may be more than an engineering trick — it may reflect something fundamental about how knowledge systems need to operate.

---

## Where This Is Now

A working prototype. The full cycle: task → navigation → agent pipeline → diff → application — works on real projects. One model (Qwen3-8B, ~5 GB VRAM), everything in-process, no external dependencies besides llama-cpp-python and tree-sitter.

The tools to build this are already available — local models on consumer hardware support everything needed. The approach can be adapted to other models and languages.

Code is open-source — [github.com/Volk-Arch/Cognit](https://github.com/Volk-Arch/Cognit).

---

*Built with Python using Qwen3-8B (transformer, GGUF quantization), tree-sitter for code navigation, and BM25 for symbol search. All operations run locally.*

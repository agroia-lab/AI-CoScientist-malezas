# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI-CoScientist is a multi-agent AI framework for collaborative scientific research, implementing the "Towards an AI Co-Scientist" paper's methodology. It uses tournament-based hypothesis evolution with Elo ratings, peer review systems, and 8 specialized agents orchestrated through the `swarms` framework.

## Commands

```bash
# Install from source (editable)
pip install -e .

# Run tests
pytest tests/

# Lint
ruff check
black --check .

# Format
black .

# Type check
mypy ai_coscientist/

# Run the example
python example.py
```

## Environment Setup

Requires one or more LLM API keys in `.env` (see `.env.example`):
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`

## Architecture

The entire framework lives in a single module: `ai_coscientist/main.py` (~2000 lines).

### Core Class: `AIScientistFramework`

Orchestrates 8 specialized `swarms.Agent` instances through a multi-phase research workflow:

1. **Generation** → produces initial hypotheses
2. **Reflection** → peer-reviews each hypothesis with structured scores (soundness, novelty, relevance, testability, clarity, impact)
3. **Ranking** → orders hypotheses by composite merit
4. **Tournament** → pairwise comparisons with Elo rating updates (`k_factor=32`, base rating `1200`)
5. **Meta-Review** → synthesizes cross-cutting insights
6. **Evolution** → refines top-k hypotheses using feedback
7. **Proximity** → clusters similar hypotheses, detects redundancy
8. Steps 2-7 repeat for `max_iterations`

### Key Data Structures (all in `main.py`)

- `Hypothesis` (dataclass) — tracks text, elo_rating, reviews, score, similarity_cluster_id, evolution_history, win/loss counts
- `WorkflowResult`, `ExecutionMetrics` (TypedDict) — returned by `run_research_workflow()`
- `HypothesisReview`, `ReviewScores`, `DetailedFeedback` (TypedDict) — structured review output
- `TournamentJudgment` — pairwise comparison result
- `AgentRole` (Enum) — the 8 agent roles

### State Persistence

`save_state()` / `load_state()` serialize agent states to `./ai_coscientist_states/` (configurable via `base_path`).

### Public API

```python
from ai_coscientist import AIScientistFramework

framework = AIScientistFramework(
    model_name="gemini/gemini-2.0-flash",
    max_iterations=3,          # 1-5
    hypotheses_per_generation=10,  # 5-20
    tournament_size=8,         # 4-12
    evolution_top_k=3,         # 2-8
    verbose=False,
)
results = framework.run_research_workflow("Your research goal here")
```

## Code Style

- **Line length: 70** (configured for both ruff and black)
- Target Python 3.10+
- Black with `preview = true`
- All agents are constructed with detailed system prompts containing explicit JSON output format instructions — changes to prompt structure require updating the corresponding JSON parsing logic

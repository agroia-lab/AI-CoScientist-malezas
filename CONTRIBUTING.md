# Contributing to AI-CoScientist

## Development Setup

1. Fork and clone the repository:

```bash
git clone https://github.com/<your-username>/AI-CoScientist.git
cd AI-CoScientist
```

2. Install in editable mode:

```bash
pip install -e .
```

3. Create a `.env` file from the example and add at least one API key:

```bash
cp .env.example .env
```

Required keys (one or more):
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GEMINI_API_KEY`

## Running Tests

```bash
pytest tests/ -v
```

Tests use mocked LLM calls and do not require API keys.

## Code Style

This project enforces strict formatting. Line length is **70 characters**.

```bash
# Format code
black --line-length 70 .

# Lint
ruff check .

# Type check
mypy ai_coscientist/
```

Configuration is in `pyproject.toml`:
- **Black**: line-length 70, target Python 3.10, `preview = true`
- **Ruff**: line-length 70
- **Python**: 3.10+

## Project Structure

```
ai_coscientist/
    __init__.py          # Package exports
    main.py              # Framework orchestrator (AIScientistFramework)
    types.py             # Data structures (Hypothesis, TypedDicts, enums)
    prompts.py           # Agent system prompts
    json_parser.py       # JSON extraction from LLM responses
    elo.py               # Elo rating logic
tests/
    conftest.py          # Shared fixtures
    test_hypothesis.py   # Hypothesis dataclass tests
    test_json_parser.py  # JSON parsing tests
    test_workflow.py     # Workflow integration tests
    test_edge_cases.py   # Edge case coverage
```

## Pull Request Process

1. Fork the repo and create a feature branch from `main`:

```bash
git checkout -b feature/your-change
```

2. Make your changes. Ensure all checks pass before pushing:

```bash
black --check .
ruff check .
mypy ai_coscientist/
pytest tests/ -v
```

3. Write descriptive commit messages that explain *why*, not just *what*.

4. Open a pull request against `main`. In the PR description, explain the
   motivation and summarize the changes.

5. All tests must pass in CI before merging.

## Architecture Notes

- **JSON output contracts**: Each agent has a system prompt specifying
  an exact JSON output format. If you change a prompt in `prompts.py`,
  you must update the corresponding parsing logic in `json_parser.py`
  (and vice versa). Mismatches will cause silent failures or crashes.

- **Public API stability**: The `AIScientistFramework` class and its
  `run_research_workflow()` method are the public interface. Changes to
  constructor parameters or return types are breaking changes and
  require discussion in an issue first.

- **Elo rating system**: Tournament comparisons use Elo ratings with
  `k_factor=32` and a base rating of `1200`. Changes to rating logic
  live in `elo.py`.

- **State persistence**: `save_state()` and `load_state()` serialize
  to `./ai_coscientist_states/`. Changes to data structures in
  `types.py` may break saved state deserialization.

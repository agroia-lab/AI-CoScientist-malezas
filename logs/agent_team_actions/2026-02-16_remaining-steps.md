# AI-CoScientist: Remaining Steps

**Date:** 2026-02-16
**Commit:** `1e7c921` — fixed 12 critical/high findings
**Baseline review:** `logs/agent_team_actions/2026-02-16_coscientist-review_report.md`

---

## What Was Fixed (Commit 1e7c921)

| # | Fix | File | Lines |
|---|-----|------|-------|
| 1 | JSON parser: stack-based brace balancer replaces broken `\{.*?\}` regex | main.py | 912-941 |
| 2 | K-factor unified to 32 (was 24 in tournament, 32 in default) | main.py | 1719 |
| 3 | API key validation at startup | main.py | 30-47 |
| 4 | State directory restricted to `0o700` | main.py | 294 |
| 5 | Null-response guard: ranking phase | main.py | 1314-1320 |
| 6 | Null-response guard: meta-review phase | main.py | 1548-1558 |
| 7 | Null-response guard: proximity phase | main.py | 1623-1633 |
| 8 | Null-response guard: tournament phase | main.py | 1755-1764 |
| 9 | Evolution preserves non-evolved hypotheses | main.py | 1910-1928 |
| 10 | `requirements.txt` includes loguru, python-dotenv | requirements.txt | 1-3 |
| 11 | Removed unused `JSONParseable` + `Protocol` import | main.py | — |
| 12 | `import re` moved to top-level (removed 3 inline imports) | main.py | 9 |

---

## Remaining Steps: Priority Order

### Phase 1 — Testing (Blocking, Do First)

**Goal:** Get from 0% to ~60% coverage on critical paths.

#### Step 1.1: Create test infrastructure
```
mkdir -p tests/
touch tests/__init__.py tests/conftest.py
```
- `conftest.py` should mock `swarms.Agent` so tests never call real LLMs
- Mock fixture: `Agent.run()` returns predefined JSON strings
- Estimated: 1 file, ~50 lines

#### Step 1.2: Unit test `_safely_parse_json`
This is the most critical function. Test cases:
- Valid JSON string → parsed dict
- JSON wrapped in markdown code fences → stripped and parsed
- Nested JSON with extra text before/after → extracted via brace balancer
- Deeply nested JSON (3+ levels) → correct extraction
- JSON with escaped quotes inside strings (`"key": "he said \"hello\""`)
- Empty string → error dict
- None input → error dict
- String with no JSON at all → error dict
- Multiple JSON objects in string → extracts first valid one
- Estimated: 1 file, ~80 lines

#### Step 1.3: Unit test `Hypothesis` dataclass
- `update_elo()` win → rating increases
- `update_elo()` loss → rating decreases
- `update_elo()` with equal ratings → symmetric update
- `to_dict()` → correct keys, win_rate calculation
- `to_dict()` with zero matches → no division by zero
- Estimated: 1 file, ~60 lines

#### Step 1.4: Integration test `run_research_workflow` with mocked agents
- Mock all 8 agents to return valid JSON
- Verify workflow completes without errors
- Verify hypotheses are generated, reviewed, ranked, evolved
- Verify evolution preserves non-evolved hypotheses
- Verify execution_metrics are populated
- Estimated: 1 file, ~120 lines

#### Step 1.5: Edge case tests
- 0 hypotheses generated → graceful fallback
- 1 hypothesis → tournament skipped
- All agents return empty → all fallbacks trigger, no crash
- Estimated: 1 file, ~80 lines

---

### Phase 2 — CI/CD Cleanup

#### Step 2.1: Audit and delete stale workflows
Delete these broken/stale files from `.github/workflows/`:
- `lints.yml` — references `pylint swarms_torch` (wrong package)
- `docs.yml` — builds docs/ directory that doesn't exist
- `docs_test.yml` — same
- `cos_integration.yml` — no integration tests exist
- Any duplicated test workflows (keep one canonical one)

#### Step 2.2: Create a working CI workflow
Create `.github/workflows/ci.yml` that runs on push/PR to main:
```yaml
- python 3.10, 3.11, 3.12
- pip install -e ".[dev]" (or poetry install)
- ruff check
- black --check --line-length 70
- pytest tests/ -v --cov=ai_coscientist
```

#### Step 2.3: Pin `swarms` dependency
In `pyproject.toml`, change:
```toml
swarms = "*"       # current — dangerous
swarms = ">=X.Y,<Z.0"  # pin to current major version
```
Check current installed version with `pip show swarms`, then pin accordingly.

---

### Phase 3 — Architecture Refactor (Medium-Term)

#### Step 3.1: Split the monolith
Extract `ai_coscientist/main.py` (2000+ lines) into modules:
```
ai_coscientist/
    __init__.py          # re-exports (keep current API)
    types.py             # Hypothesis, TypedDicts, AgentRole
    prompts.py           # 8 prompt methods → functions
    json_parser.py       # _safely_parse_json (now independently testable)
    elo.py               # Elo rating logic
    framework.py         # AIScientistFramework (orchestrator only)
```
**Rule:** No changes to public API. `from ai_coscientist import AIScientistFramework` must still work.

#### Step 3.2: Make prompts configurable
Allow loading prompts from files or constructor kwargs:
```python
AIScientistFramework(
    generation_prompt="path/to/custom_prompt.txt",  # or string
)
```
Fall back to built-in defaults when not provided.

#### Step 3.3: Add agent abstraction layer
Define a protocol/ABC so the framework isn't locked to `swarms.Agent`:
```python
class AgentInterface(Protocol):
    def run(self, input: str) -> str: ...
    def save_state(self) -> None: ...
    def load_state(self) -> None: ...
```

---

### Phase 4 — Robustness (Medium-Term)

#### Step 4.1: Add workflow quality gates
After each phase, check output quality. Stop early if degraded:
```python
if len(self.hypotheses) == 0:
    raise WorkflowDegradedError("Generation produced 0 hypotheses")
if all(h.score == 0.0 for h in self.hypotheses):
    logger.warning("All hypotheses scored 0.0 — results unreliable")
```

#### Step 4.2: Improve tournament statistical validity
- Option A: Round-robin (every pair compared once) for small N
- Option B: Swiss-system for larger pools
- Add reproducibility via `random.seed()`

#### Step 4.3: Bound conversation history
Add a max-size or pruning strategy so memory doesn't grow unbounded:
```python
if len(self.conversation) > MAX_HISTORY:
    self.conversation.prune(keep_last=MAX_HISTORY)
```

---

### Phase 5 — Documentation (Low Priority)

#### Step 5.1: Fix README examples
- Change `from ai_scientist import CoScientist` to `from ai_coscientist import AIScientistFramework`
- Update default model_name from "gpt-4" to "gpt-4.1"

#### Step 5.2: Add CONTRIBUTING.md
- Dev setup instructions
- How to run tests
- Code style (70-char line length, black, ruff)
- PR process

#### Step 5.3: Add `.gitignore` entries
```
.env
*.log
.mypy_cache/
.ruff_cache/
dist/
build/
```

---

## Effort Estimates

| Phase | Steps | Effort | Impact |
|-------|-------|--------|--------|
| 1. Testing | 5 steps | 2-3 hours | HIGH — enables safe future changes |
| 2. CI/CD | 3 steps | 1 hour | HIGH — prevents regressions |
| 3. Architecture | 3 steps | 4-6 hours | MEDIUM — improves maintainability |
| 4. Robustness | 3 steps | 2-3 hours | MEDIUM — improves reliability |
| 5. Documentation | 3 steps | 1 hour | LOW — reduces onboarding friction |

**Recommended order:** Phase 1 → Phase 2 → Phase 5 → Phase 3 → Phase 4

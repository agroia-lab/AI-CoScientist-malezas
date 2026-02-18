# Consolidated Roadmap: AI-CoScientist for Agronomy & Precision Farming
**Created:** 2026-02-18
**Status:** Replaces and extends `NEXT_STEPS.md`
**Baseline:** 132 tests passing, `gpt-4.1-mini`, pipeline validated end-to-end

---

## Sources Synthesized

| Source | What we took |
|--------|-------------|
| Original `NEXT_STEPS.md` | 10 agreed items (all carried forward, re-prioritized) |
| SakanaAI/AI-Scientist v1+v2 | Iterative self-reflection, ensemble reviews, novelty API checking, tree-search evolution |
| assafelovic/gpt-researcher | Chief-editor orchestration, parallel subquestion research, source credibility scoring |
| lfleon9b/gpt-researcher (fork) | Domain-adapted research planning patterns |
| Stanford STORM / Co-STORM | Perspective-driven generation, simulated dialogue review, shared mind-map |
| Microsoft AutoGen | Dynamic agent routing, AgentTool pattern, convergence-based termination |
| OpenScholar / PaperQA2 | Literature-grounded hypothesis validation, citation verification |
| ChemCrow (UR White Lab) | Domain tool integration pattern (databases, APIs, calculators as agent tools) |
| AgentSquare / MoLAS | Module evolution for auto-improving agent prompts |
| AgriBench + agronomy AI survey | Agronomy-specific evaluation criteria, multimodal grounding |

---

## Current Architecture (reference)

8 fixed-phase agents → `Generation → Reflection → Ranking → Tournament → [Meta-Review → Evolution → Reflection → Ranking → Tournament → Proximity] × N`

11 review criteria: 6 generic (soundness, novelty, relevance, testability, clarity, impact) + 5 agronomy-specific (statistical_rigor, field_feasibility, spatial_scalability, environmental_sustainability, agronomic_practicality).

Known open issues: hypothesis wordiness (W3), score compression 0.72–0.87 (W4), high-tech portfolio bias (W7), meta-review insights not fed back (W6), Elo noisy at small N (W1).

---

## TIER 1 — Prompt Changes Only (highest ROI, do now)

These touch only `prompts.py` and require no architectural changes.

### T1-A · Hypothesis Parsimony Constraint *(from NEXT_STEPS #2)*
**Problem:** Evolution produces 300-word grant proposals instead of testable 2–3 sentence claims.
**Fix:** Add to evolution prompt:
> *"Reduce the hypothesis to its single most testable core claim. One independent variable, one dependent variable, one mechanism. Remove implementation details that belong in a research protocol, not a hypothesis. Target: 2–3 sentences, ≤60 words."*

**Expected output:**
> *"Sequential priming with Nijmegen-1 (10⁻⁷ M) followed by F. oxysporum f.sp. orthoceras (10⁸ cfu mL⁻¹) reduces P. ramosa seed bank more than either treatment alone, because suicidal germination exposes the radicle before tubercle formation."*

---

### T1-B · Adversarial Review Pass *(from NEXT_STEPS #3)*
**Problem:** Reflection agent scores everything 4–5/5; no meaningful differentiation.
**Fix:** Add a second reflection agent with sceptical framing:
> *"You are a sceptical reviewer whose job is to find flaws, not confirm merit. Identify the single most likely reason this hypothesis would fail in a Mediterranean field trial. Score each criterion assuming maximum scrutiny."*

Run both passes per hypothesis. Final score = average of optimistic + adversarial reviewer. This mechanically forces spread without changing the 11-criteria schema.

---

### T1-C · Portfolio Balance Constraints *(from NEXT_STEPS #4)*
**Problem:** Portfolio clusters on high-tech solutions (UAV, ML, VRT). Missing: trap crops, rotation, resistant varieties.
**Fix:** Add mandatory coverage rules to supervisor prompt:

- ≥1 hypothesis implementable without specialised equipment (rotation, hand-weeding, variety selection)
- ≥1 hypothesis addressing cultural/preventive management (trap crops, non-host rotation, irrigation timing)
- ≥1 hypothesis involving biological control (biocontrol agents, allelopathic cover crops)
- Technology-focused hypotheses ≤ 40% of portfolio

*P. ramosa specific:* ensure coverage of linseed/flax trap crops, Clearfield imidazolinone-resistant varieties, non-host rotation (maize, cereals).

---

### T1-D · Self-Scoring at Generation Time *(from AI-Scientist)*
**Problem:** All N hypotheses enter the full review pipeline regardless of quality, wasting API calls.
**Fix:** After generation, each hypothesis self-scores on 4 dimensions (1–10):
- Interestingness / scientific novelty
- Field feasibility (can a Mediterranean grower do this?)
- Testability (can be designed as a replicated trial?)
- Parsimony (is it a single testable claim?)

Hypotheses scoring below threshold (e.g., mean < 5) are discarded before entering reflection. This pre-filters ~20–30% of hypotheses cheaply.

---

### T1-E · Structured Review Format *(from AI-Scientist)*
**Problem:** Reviews are scores only; no structured rationale for evolution to use.
**Fix:** Extend reflection output to include per-hypothesis:
- `strengths`: list of 2–3 specific merits
- `weaknesses`: list of 2–3 specific flaws
- `killer_question`: the one empirical question that would confirm or refute this hypothesis
- `fail_scenario`: most likely reason this fails in a Mediterranean field trial (feeds adversarial pass)

Evolution agent reads `weaknesses` and `killer_question` directly as refinement targets.

---

### T1-F · Perspective-Driven Generation *(from STORM)*
**Problem:** Generator produces hypotheses from a blank slate, missing systematic angle coverage.
**Fix:** Before generation, add a "perspective discovery" micro-step to the supervisor:

1. Enumerate 6–8 research angles from the literature structure of the problem (e.g., "chemical priming", "cultural exclusion", "genetic resistance", "biological suppression", "remote sensing for early detection", "decision support for spray timing")
2. Assign ≥1 hypothesis to each perspective before generation begins

This is STORM's core insight applied to hypothesis space instead of document space. It ensures the portfolio is systematically constructed rather than randomly sampled.

---

### T1-G · Strengths-Weighted Tournament Framing *(from gpt-researcher chief-editor pattern)*
**Problem:** Tournament judge picks a winner on 4 grouped dimensions but framing is neutral.
**Fix:** Frame tournament as adversarial debate:
- H_A: argue for your hypothesis, cite `strengths` from review
- H_B: argue for your hypothesis, cite `strengths` from review
- Judge evaluates which argument is more convincing given Mediterranean farming constraints

This produces more discriminating judgments than symmetric comparison and aligns with how expert panels actually evaluate field research proposals.

---

## TIER 2 — Code Changes Required (high impact, next sprint)

These require changes to `main.py` and/or `types.py`.

### T2-A · Ensemble Review Aggregation *(from AI-Scientist)*
**Problem:** Single-pass review is unreliable; one bad review can distort Elo.
**Fix:** Run reflection agent **3 times per hypothesis** (temperature=0.75 for diversity). Average the 11 numerical scores across runs. Add a meta-reviewer (6th call) that reads all 3 reviews and synthesizes conflicting assessments into a coherent narrative.

Implementation in `main.py`:
```python
def _review_hypothesis(self, h: Hypothesis) -> HypothesisReview:
    reviews = [self._reflection_agent.run(task) for _ in range(3)]
    # + adversarial review (T1-B)
    averaged = _average_scores(reviews)
    meta_narrative = self._meta_reflection_agent.run(reviews)
    return HypothesisReview(scores=averaged, narrative=meta_narrative)
```

---

### T2-B · Directed Regeneration from Meta-Review *(from NEXT_STEPS #5)*
**Problem:** Meta-review identifies portfolio gaps but generation never responds to them — the loop doesn't close.
**Fix:** After meta-review, supervisor parses the gap analysis and issues **targeted generation** calls:

```
meta_review identifies: "no hypothesis covers cultural exclusion via trap crops"
→ supervisor: "Generate 2 new hypotheses for [linseed trap crop / non-host rotation]"
→ new hypotheses enter the pool alongside existing ones
→ tournament compares new vs. existing via Elo
```

This closes the iterative improvement loop and is the single most impactful architectural fix.

---

### T2-C · Multi-Criterion Tournament Scoring *(from NEXT_STEPS #6)*
**Problem:** Single winner-pick is noisy at small N; doesn't distinguish field-practical from scientifically elegant.
**Fix:** Replace binary winner with per-dimension scoring:

| Dimension | H_A | H_B |
|-----------|-----|-----|
| Scientific soundness | 4 | 3 |
| Field realism | 3 | 4 |
| Novelty | 4 | 4 |
| Farmer accessibility | 2 | 4 |

Each dimension gets its own Elo rating. Final ranking uses weighted sum:
- Field realism × 0.30
- Scientific soundness × 0.25
- Novelty × 0.20
- Farmer accessibility × 0.15
- Environmental sustainability × 0.10

Weights configurable per research context (e.g., for a purely academic study, increase scientific soundness weight).

---

### T2-D · Novelty Checking via Literature API *(from AI-Scientist)*
**Problem:** Hypotheses are generated from LLM parametric knowledge with no check against published literature.
**Fix:** After generation and self-scoring (T1-D), add a **novelty checker** step that queries Semantic Scholar or OpenAlex:

```python
def _check_novelty(self, hypothesis_text: str) -> NoveltyResult:
    queries = self._generate_search_queries(hypothesis_text)  # LLM step
    papers = semantic_scholar_search(queries)  # API call
    overlap = self._assess_overlap(hypothesis_text, papers)  # LLM step
    return NoveltyResult(is_novel=overlap < 0.7, references=papers)
```

Hypotheses flagged as non-novel get a novelty penalty in scoring, or are regenerated with an instruction to differentiate from the overlapping papers.

APIs: Semantic Scholar (free, 100 req/5min), OpenAlex (free, no key needed).

---

### T2-E · Convergence-Based Evolution Termination *(from AutoGen)*
**Problem:** Evolution runs fixed `max_iterations` even if hypotheses have plateaued.
**Fix:** Add convergence detection — stop iterating when:
- Top-k hypothesis texts change < 15% (Jaccard similarity between iterations)
- Composite review scores change < 0.05 between iterations
- OR `max_iterations` is reached (existing ceiling)

This prevents wasted API calls on stagnant runs and also surfaces "stuck" scenarios for human intervention.

---

### T2-F · Iterative Self-Refinement in Generation *(from AI-Scientist)*
**Problem:** Generation is single-shot; quality depends entirely on one prompt execution.
**Fix:** After initial generation, run a self-reflection micro-loop per hypothesis (up to 3 rounds):
1. LLM generates hypothesis
2. LLM self-critiques: *"Does this meet parsimony? Is it falsifiable? Does it map to a specific IWM pillar?"*
3. If critique finds issues → refine and repeat (up to 3 rounds)
4. If critique approves → proceed to self-scoring (T1-D)

This is cheap (same model, cheap self-critique call) and produces notably sharper initial hypotheses before peer review.

---

### T2-G · Parallel Subquestion Research *(from gpt-researcher)*
**Problem:** Supervisor generates a research plan but ignores existing literature on sub-problems.
**Fix:** Before hypothesis generation, run a **research phase** inspired by gpt-researcher's planner+researcher pattern:

1. Supervisor decomposes goal into 4–6 research subquestions (e.g., "What germination stimulants are effective against Phelipanche? What biological agents have been field-tested in Mediterranean climates?")
2. Researcher agent runs parallel searches (Semantic Scholar, OpenAlex) per subquestion
3. Summaries fed into Generation prompt as `[LITERATURE CONTEXT]`

Result: hypotheses are grounded in real published evidence rather than LLM parametric knowledge. Cost: ~6 API calls + literature search.

---

## TIER 3 — Architecture Changes (medium-term)

### T3-A · Dynamic Agent Routing *(from AutoGen SelectorGroupChat)*
Replace the fixed-phase pipeline with an LLM-based selector that chooses the next agent based on conversation state. Example: if a hypothesis has very low field_feasibility score, route directly to a "field-expert" specialist agent instead of generic evolution.

---

### T3-B · Hypothesis Knowledge Graph *(from STORM Co-STORM)*
Track relationships between hypotheses across iterations as a graph (hypothesis nodes, edges for: "evolved from", "contradicts", "complements", "same IWM pillar"). Use graph structure to:
- Identify over-represented clusters and prompt regeneration for sparse regions
- Surface "synthesis hypotheses" that combine complementary approaches

---

### T3-C · Simulated Dialogue Review *(from STORM Co-STORM)*
Replace one-shot reflection with a simulated conversation: hypothesis author defends their claim, reviewer challenges it, author responds to objections. Final review scores emerge from the dialogue. Produces richer `weaknesses` and `killer_question` fields (T1-E).

---

### T3-D · Agent-as-Tool Pattern *(from AutoGen)*
Let the supervisor invoke any agent as a tool on-demand rather than waiting for the fixed phase schedule. E.g., supervisor can call proximity analysis mid-generation if it suspects redundancy, or call novelty-checker mid-evolution if a hypothesis diverges into known territory.

---

### T3-E · Branching Evolution via Tree Search *(from AI-Scientist v2 BFTS)*
Instead of linearly refining top-k hypotheses, maintain an **evolution tree**:
- Each hypothesis can branch into multiple variants (e.g., "mechanistic branch", "practical branch", "scale-up branch")
- Variants are scored and pruned by the ranking agent
- Best branch per hypothesis advances to tournament

This mirrors how actual scientific development works (exploring multiple experimental directions before committing).

---

## TIER 4 — Future / Exploratory

### T4-A · Domain Tool Integration *(from ChemCrow pattern)*
Add callable domain tools to agents:
- **Germplasm database**: query resistant tomato varieties (Clearfield system)
- **GBIF / EPPO data**: retrieve known Phelipanche host range and distribution
- **FAO GAEZ**: retrieve Mediterranean climate/soil data for feasibility checks
- **Sentinel Hub API**: retrieve field-level NDVI for spatial grounding

This transforms hypothesis generation from text-only to evidence-grounded reasoning.

---

### T4-B · Multimodal Grounding *(from AgriBench / agronomy AI survey)*
Feed field imagery (RGB, NDVI, thermal) and soil maps as context to generation and reflection agents. Ground hypotheses in observed field patterns rather than purely abstract reasoning. Relevant given existing `ndvi-orara-spatial-workflow` skill in the local environment.

---

### T4-C · Auto-Evolving Agent Prompts *(from AgentSquare MoLAS)*
Decompose each agent into Planning + Reasoning + Tool + Memory modules. After each iteration, run a meta-optimization step that evaluates module configurations and generates improved variants for the next iteration. Long-term: the framework improves its own agents over time based on hypothesis quality metrics.

---

### T4-D · Adaptive Tournament Difficulty *(from nonstationary benchmarks)*
As hypotheses improve across iterations, automatically raise the bar for tournament win criteria. Early iterations: any field-feasible hypothesis can win. Later iterations: winner must also be novel relative to published literature AND have a defined statistical design.

---

## Infrastructure & Housekeeping

| ID | Task | Priority | Notes |
|----|------|----------|-------|
| I1 | Fix GitHub push | **High** | Investigate credentials/SSH/token |
| I2 | Validate at scale (N=10, iter=2) | **High** | After Tier 1 prompts done |
| I3 | Conditional activation (skip supervisor/proximity at N<10) | Low | ~$0.01 savings per run |
| I4 | `save_state()` for DirectLLMAgent | Low | Reproducibility |
| I5 | Claude compatibility | Low | gpt-4.1-mini works; low priority |

---

## Suggested Implementation Order

```
Sprint 1 (prompt-only, 1 session):
  T1-A  Parsimony constraint in evolution
  T1-B  Adversarial review pass (new reflection agent)
  T1-C  Portfolio balance constraints in supervisor
  T1-D  Self-scoring at generation time
  T1-E  Structured review format (strengths/weaknesses/killer_question)
  T1-F  Perspective-driven generation (supervisor micro-step)
  I1    Fix GitHub push
  → Validate: run example.py, check hypothesis quality, score spread

Sprint 2 (code changes, 1-2 sessions):
  T2-A  Ensemble review aggregation (3× reflection + meta-reviewer)
  T2-B  Directed regeneration from meta-review (closes the loop)
  T2-C  Multi-criterion tournament scoring (per-dimension Elo)
  I2    Validate at scale (N=10, iter=2)

Sprint 3 (literature grounding, 1 session):
  T2-D  Novelty checking via Semantic Scholar / OpenAlex
  T2-G  Parallel subquestion research before generation
  T2-F  Iterative self-refinement in generation
  T2-E  Convergence-based evolution termination

Sprint 4+ (architecture, multi-session):
  T3-A  Dynamic agent routing
  T3-B  Hypothesis knowledge graph
  T3-C  Simulated dialogue review
  T3-D  Agent-as-tool pattern
  T3-E  Branching evolution tree search

Future:
  T4-A  Domain tool integration
  T4-B  Multimodal grounding
  T4-C  Auto-evolving agent prompts
  T4-D  Adaptive tournament difficulty
```

---

## Priority Matrix

| Item | Scientific Impact | Practical Farming Value | Implementation Cost | ROI |
|------|------------------|------------------------|--------------------|----|
| T1-A Parsimony | ★★★★★ | ★★★★☆ | ★☆☆☆☆ | **★★★★★** |
| T1-B Adversarial review | ★★★★☆ | ★★★☆☆ | ★☆☆☆☆ | **★★★★★** |
| T1-C Portfolio balance | ★★★☆☆ | ★★★★★ | ★☆☆☆☆ | **★★★★★** |
| T1-F Perspective-driven gen | ★★★★☆ | ★★★★☆ | ★☆☆☆☆ | **★★★★★** |
| T2-B Directed regeneration | ★★★★★ | ★★★★☆ | ★★★☆☆ | **★★★★☆** |
| T2-A Ensemble reviews | ★★★★☆ | ★★★☆☆ | ★★☆☆☆ | **★★★★☆** |
| T2-C Multi-criterion Elo | ★★★★☆ | ★★★★☆ | ★★★☆☆ | **★★★★☆** |
| T2-D Novelty checking | ★★★★★ | ★★★☆☆ | ★★★☆☆ | **★★★★☆** |
| T2-G Parallel lit research | ★★★★★ | ★★★★☆ | ★★★☆☆ | **★★★★☆** |
| T3-A Dynamic routing | ★★★☆☆ | ★★★☆☆ | ★★★★☆ | **★★★☆☆** |
| T3-E Evolution tree search | ★★★★☆ | ★★☆☆☆ | ★★★★★ | **★★☆☆☆** |
| T4-A Domain tools | ★★★★★ | ★★★★★ | ★★★★★ | **★★★★☆** |
| T4-B Multimodal grounding | ★★★★☆ | ★★★★★ | ★★★★★ | **★★★☆☆** |

---

## Reference

- Baseline issues: `DEBUGGING_REPORT.md`, `ISSUES_AND_WORKFLOW_REVIEW.md`
- Original 10-item plan: `NEXT_STEPS.md` (superseded by this document)
- Current working config: `gpt-4.1-mini`, `example.py`, `max_iterations=1`, N=5

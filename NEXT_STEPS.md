# Next Steps
**Last updated:** 2026-02-18
**Current state:** Pipeline working end-to-end with `gpt-4.1-mini`, 132 tests passing,
committed at `35cc127`.

---

## 1. Fix GitHub Push Issue
**Priority: High — blocks sharing and collaboration**

- Investigate recurring push failures (credentials, remote config, or branch protection)
- Verify `git remote -v`, SSH vs HTTPS, token expiry
- Document the fix

---

## 2. Sharpen Hypothesis Definition
**Priority: High — affects scientific value of all outputs**

The evolution prompt currently makes hypotheses longer and more elaborate.
Add a parsimony constraint so evolution produces a focused, single-claim statement:

> *"Reduce the hypothesis to its single most testable core claim. One independent
> variable, one dependent variable, one mechanism. Remove implementation details
> that belong in a research protocol, not a hypothesis. Target: 2–3 sentences."*

Expected outcome: evolution output goes from 300-word research proposals to
crisp 3-sentence hypotheses like:
> *"Sequential application of Nijmegen-1 (10⁻⁷ M, 14 days before) followed by
> F. oxysporum f.sp. orthoceras (10⁸ cfu mL⁻¹, 7 days after) reduces P. ramosa
> seed bank more than either treatment alone, because suicidal germination exposes
> the radicle before tubercle formation."*

---

## 3. Add Adversarial Review Pass
**Priority: High — scores currently cluster too narrow (0.72–0.87)**

Add a second reflection pass with an explicitly critical framing:

> *"You are a sceptical reviewer whose job is to find flaws, not confirm merit.
> Identify the single most likely reason this hypothesis would fail in a
> Mediterranean field trial."*

This creates score differentiation and surfaces weak assumptions before tournament.

---

## 4. Portfolio Balance Constraint in Supervisor
**Priority: High — current output misses key IWM approaches**

Add mandatory coverage rules to the supervisor prompt:

- At least 1 hypothesis must be implementable **without specialised equipment**
  (crop rotation, variety selection, manual removal)
- At least 1 hypothesis must address **cultural/preventive management**
  (trap crops, non-host rotation, irrigation timing)
- At least 1 hypothesis must involve **biological control**
  (biocontrol agents, allelopathic cover crops)
- Technology-focused hypotheses ≤ 40% of the portfolio

For *P. ramosa* specifically, ensure coverage of:
- Linseed / flax trap crops (well-established, low-tech)
- Imidazolinone-resistant tomato varieties (Clearfield system)
- Non-host rotation (maize, cereals)

---

## 5. Directed Regeneration from Meta-Review
**Priority: Medium — needed for true iterative improvement**

Current behaviour: meta-review identifies gaps → insights passed to evolution → evolution ignores them.

New behaviour: after meta-review, supervisor reads the gap analysis and issues
a **targeted generation call** for the missing IWM pillars:

```
if meta_review identifies uncovered pillars:
    → supervisor: "Generate 2 new hypotheses for [missing pillar]"
    → new hypotheses enter the pool alongside existing ones
    → tournament compares new vs. existing
```

---

## 6. Multi-Criterion Tournament Scoring
**Priority: Medium — Elo is too noisy at small N**

Replace single "pick a winner" pairwise judgment with a head-to-head scoring table:

| Dimension | H_A | H_B |
|-----------|-----|-----|
| Scientific soundness | 4 | 3 |
| Field realism | 3 | 4 |
| Novelty | 4 | 4 |
| Farmer accessibility | 2 | 4 |

Elo updated per dimension, giving a **multi-dimensional Elo profile** per hypothesis.
Final ranking uses weighted sum (weights configurable per research context).

---

## 7. Validate at Larger Scale
**Priority: Medium — test if fixes hold under realistic load**

After items 2–6 are implemented, run:

```python
AIScientistFramework(
    model_name="gpt-4.1-mini",
    max_iterations=2,
    hypotheses_per_generation=10,
    tournament_size=8,
    evolution_top_k=3,
)
```

Evaluate:
- Are the top hypotheses now crisp and focused (not 300-word proposals)?
- Is there meaningful score spread (e.g., 0.4–0.9 range)?
- Does directed regeneration fill portfolio gaps?
- Does the loop genuinely improve across iteration 1 → 2?

---

## 8. Conditional Activation of Lightweight Steps
**Priority: Low — optimisation only**

- Supervisor: only activate when `hypotheses_per_generation >= 10`
- Proximity analysis: only activate when `N >= 10`

Saves ~25–30 seconds and ~$0.01 per small run.

---

## 9. Claude Compatibility Investigation
**Priority: Low — gpt-4.1-mini works well**

If Anthropic access is important:
- Open a support ticket with Anthropic to whitelist agricultural research domain
- Or: test `claude-3-5-sonnet` older models before content policy tightening
- Or: use Claude only for non-IWM-content agents (meta-review, proximity) + GPT for generation/reflection/tournament

---

## 10. `save_state()` for DirectLLMAgent
**Priority: Low — not needed for single runs**

Implement state persistence for reproducibility:
- Save `(model_name, system_prompt, llm_args)` to JSON
- Enable resuming interrupted workflows

---

## Suggested Order of Work (Next Session)

```
1. Fix GitHub push
2. Items 2 + 3 + 4  (prompt changes — can be done in parallel)
3. Item 5            (directed regeneration — requires main.py change)
4. Item 7            (validation run)
5. Items 6, 8, 9, 10 (if time permits)
```

---

## Reference

- Technical issues + root causes: `DEBUGGING_REPORT.md`
- Conceptual workflow problems: `ISSUES_AND_WORKFLOW_REVIEW.md`
- Scientific quality of current hypotheses: see inline review in `ISSUES_AND_WORKFLOW_REVIEW.md` Part II
- Current working config: `gpt-4.1-mini`, `example.py`, `max_iterations=1`, `N=5`

# Feature direction is read from the dependence slope, not from mean SHAP

Two of the deposited figures — `fig02_consensus_signatures` and
`supp_fig06_feature_selection_frequency` — colour each bar by the **sign of the
feature's mean SHAP value**. That statistic does not carry a direction, and this
note records why, what is used instead, and how far apart the two are.

## Why mean SHAP cannot carry a direction

SHAP attributions are defined against a baseline, so a feature's SHAP values
**average to approximately zero over the sample by construction**. What survives
the averaging is not the effect; it is the residual asymmetry of the sample,
which is noise. In this analysis `|mean SHAP|` is a small fraction of
`mean|SHAP|` at the median — the signal is in the spread, not in the mean.

The sign of that residual was then printed as if it were the direction of the
biology.

## What is used instead

The direction of a feature is the **SHAP dependence slope**: the correlation
between the feature's within-fold standardised value and its own SHAP value,
pooled over the folds that selected it. "+" means a higher value of the feature
pushes the prediction towards pCR. This is the quantity a beeswarm plot encodes
in its colour axis, and it is what `fig_feature_ranking_by_scope` uses.

## How far apart the two are

Measured over all **63** bars of the consensus signatures:

| Comparison | Agreement |
|---|---|
| mean-SHAP sign vs the dependence slope | **29 / 63 (46%)** — no better than a coin flip |

**34 of the 63 bars would carry the opposite sign** under the two statistics.
That is not a marginal disagreement to be noted in passing; it is why the two
figures above should be read for magnitude only.

## Bars whose direction is not stable

A direction that reverses between folds should not be quoted even when the
pooled slope is clear. Where a feature's per-fold slope agrees with its pooled
slope in fewer than **75%** of the folds that selected it, the bar keeps its
sign but is drawn in parentheses and greyed. On this analysis that brackets
**8 of the 63 bars**.

The threshold is `--min-sign-stability` in `make_fig_feature_ranking.py`; pass 0
to bracket nothing. Per-bar values — `shap_dependence_r`,
`sign_stability_frac_folds` and `direction_low_stability` — are tabulated in
`report/tables/revision/fig_feature_ranking_by_scope.xlsx`, so every sign in the
figure can be checked against the number behind it.

## What to read where

| Question | Figure |
|---|---|
| How strongly does a feature rank within its modality? | `fig02`, `supp_fig06` — bar length |
| Which direction does it push? | `fig_feature_ranking_by_scope` — and its workbook |

`fig02` and `supp_fig06` are kept because their **magnitudes** are correct and
they are the panels the manuscript's figures derive from. Their colouring is
not, and neither figure should be used to state a direction.

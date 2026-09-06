#!/usr/bin/env python3
"""
Outcome-blind feature deduplication, applied once per scenario before any split.

WHY THIS EXISTS
---------------
The pipeline's redundancy control was a single correlation threshold at
|r| = 0.90, enforced across the pooled cohort. It is coherent and its contract
holds, but it does not catch three things that matter at this sample size:

  1. Near-duplicates in continuous features. The ER/luminal axis appears three
     times in the T-DM1 transcriptomic signature (|r| up to 0.864), so five
     features carry two independent quantities.

  2. Segment-called copy number. Genes in one called segment receive ONE number.
     DNA_ERBB2_CNA and DNA_RPL19_CNA are the identical value in 73% of patients
     and correlate at only 0.583, because the scale is capped and the discordant
     minority drags the coefficient down. No correlation threshold anyone would
     accept catches that pair; the right statistic is how often the two columns
     agree exactly.

  3. Population dependence. DNA_coding_mutation_HER_pathway and
     DNA_coding_mutation_ERBB2_oncokb are r = 1.000 WITHIN THE DHP ARM and only
     borderline pooled. Collinearity is a property of a population, so the rule
     is applied within each scenario's own cohort.

THE TRAP THIS AVOIDS
--------------------
Applying the identical-value test to sparse binary indicators is wrong. Two
mutations each present in 5% of patients agree in ~90% of rows by rarity alone:
DNA_coding_mutation_GATA3 and DNA_coding_mutation_ERBB2 agree in 90% of patients
and NEVER co-occur once. Judged by agreement they look like duplicates; judged by
Cohen's kappa, which corrects for prevalence, kappa = -0.05. So binary features
are compared by kappa, never by raw agreement.

WHAT MAKES IT OUTCOME-BLIND, AND WHY THAT MATTERS
-------------------------------------------------
No statistic here reads pCR. Correlation, exact agreement and kappa are computed
between FEATURES only. That is what makes it legitimate to run before splitting:
a pre-split step that consulted the outcome would be precisely the leak this
revision exists to remove. A per-fold filter is the alternative, and it is worse
here for a separate reason -- it keeps whichever cluster member wins that fold's
contest, so the survivor rotates between folds and a pair of near-identical
features splits its selection frequency and BOTH reach the consensus signature.

WITHIN MODALITY, NOT ACROSS
---------------------------
Fusion is at the level of predictions: each modality is modelled separately and
only the second-stage combiner sees all five. An RNA feature and a DNA feature
therefore never enter the same design matrix, so cross-modality correlation is
not collinearity for any fitted model. RNA_HER2DX_HER2_amplicon correlates with
DNA_ERBB2_CNA at 0.816 in the DHP arm -- real biology, no collinearity. Only
within-modality pairs are compared.
"""
from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# --------------------------------------------------------------- thresholds --
CORR_T = 0.80      # |r|, continuous features
IDENT_T = 0.80     # exact-agreement fraction, segment-called CNA
KAPPA_T = 0.80     # Cohen's kappa, binary / sparse indicators
MIN_N = 20         # a pair needs this many jointly observed patients to judge

# Which member of a cluster to keep. Biology, decided in advance, never the
# outcome. Earlier entries win. Anything not listed falls through to the
# deterministic tie-break in _representative().
PREFER_KEEP = [
    "DNA_ERBB2_CNA",                  # the canonical 17q driver; MED1 and RPL19
                                      # are passengers on the same amplicon
    "DNA_coding_mutation_PIK3CA",     # the named gene over the pathway roll-up
    "DNA_coding_mutation_ERBB2",
    "DNA_coding_mutation_TP53",
]
# Deprioritised: aggregates and annotation-derived copies are passengers of the
# column they summarise, so they lose a tie to it.
PREFER_DROP_SUFFIX = ("_pathway", "_oncokb")


@dataclass
class Decision:
    scenario: str
    modality: str
    kept: str
    dropped: str
    statistic: str
    value: float
    n: int


@dataclass
class DedupResult:
    keep: list[str]
    drop: list[str]
    decisions: list[Decision] = field(default_factory=list)
    clusters: list[list[str]] = field(default_factory=list)


def _is_binary(v: np.ndarray) -> bool:
    u = np.unique(v[~np.isnan(v)])
    return len(u) <= 2 and set(u.tolist()) <= {0.0, 1.0}


def _is_segment_called(name: str, v: np.ndarray) -> bool:
    """Segment-called copy number: a CNA column with enough distinct values to be
    continuous. The name alone is not enough -- a binary CNA call is judged as a
    binary feature -- and continuity alone is not enough either, because exact
    agreement is only meaningful where one segment call feeds several genes."""
    return name.endswith("_CNA") and len(np.unique(v[~np.isnan(v)])) > 10


def cohens_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Agreement beyond what the two prevalences predict."""
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def _pair_verdict(name_a, name_b, xa, xb):
    """-> (redundant, statistic_name, value) using the statistic the data type
    warrants. One statistic cannot serve all three types; using the wrong one is
    how the 17q passengers survived and how the rare mutations nearly did not."""
    if _is_binary(xa) and _is_binary(xb):
        k = cohens_kappa(xa, xb)
        return (k >= KAPPA_T, "kappa", k)
    if _is_segment_called(name_a, xa) and _is_segment_called(name_b, xb):
        ident = float((xa == xb).mean())
        if ident >= IDENT_T:
            return (True, "identical", ident)
    if xa.std() == 0 or xb.std() == 0:
        return (False, "constant", float("nan"))
    r = abs(float(np.corrcoef(xa, xb)[0, 1]))
    return (r >= CORR_T, "|r|", r)


def _representative(cluster: list[str], frame: pd.DataFrame) -> str:
    """Which member survives. Deterministic and outcome-blind, in order:
    an explicit biological preference; then not being an aggregate or an
    annotation copy; then the most completely measured column; then
    alphabetical, so the result never depends on dict ordering."""
    for want in PREFER_KEEP:
        if want in cluster:
            return want
    plain = [c for c in cluster if not c.endswith(PREFER_DROP_SUFFIX)]
    pool = plain or cluster
    return sorted(pool, key=lambda c: (-int(frame[c].notna().sum()), c))[0]


def deduplicate(frame: pd.DataFrame, columns: list[str],
                scenario: str = "") -> DedupResult:
    """Deduplicate `columns` within `frame`, within modality.

    `frame` must already be restricted to the scenario's own patients: the rule
    is population-specific by design.
    """
    X = frame[columns].apply(pd.to_numeric, errors="coerce")
    usable = [c for c in X.columns if X[c].notna().sum() >= MIN_N]
    by_mod: dict[str, list[str]] = defaultdict(list)
    for c in usable:
        by_mod[c.split("_", 1)[0]].append(c)

    adj: dict[str, set[str]] = defaultdict(set)
    found: list[tuple[str, str, str, float, int]] = []
    for mod, cols in by_mod.items():
        for a, b in itertools.combinations(sorted(cols), 2):
            both = X[[a, b]].dropna()
            if len(both) < MIN_N:
                continue
            xa, xb = both[a].to_numpy(float), both[b].to_numpy(float)
            red, stat, val = _pair_verdict(a, b, xa, xb)
            if red:
                adj[a].add(b)
                adj[b].add(a)
                found.append((a, b, stat, val, len(both)))

    seen: set[str] = set()
    clusters: list[list[str]] = []
    for node in sorted(adj):
        if node in seen:
            continue
        stack, comp = [node], []
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            comp.append(cur)
            stack.extend(sorted(adj[cur] - seen))
        clusters.append(sorted(comp))

    drop: list[str] = []
    decisions: list[Decision] = []
    for cluster in clusters:
        rep = _representative(cluster, X)
        for member in cluster:
            if member == rep:
                continue
            drop.append(member)
            # Report the pair that justifies the removal: the strongest link
            # between the dropped member and the representative if they are
            # directly linked, otherwise its strongest link inside the cluster.
            direct = [f for f in found if {f[0], f[1]} == {member, rep}]
            link = direct or [f for f in found if member in (f[0], f[1])]
            _, _, stat, val, n = max(link, key=lambda f: abs(f[3]))
            decisions.append(Decision(scenario, member.split("_", 1)[0],
                                      rep, member, stat, float(val), n))

    keep = [c for c in columns if c not in set(drop)]
    return DedupResult(keep=keep, drop=sorted(drop),
                       decisions=decisions, clusters=clusters)


def audit_frame(result: DedupResult) -> pd.DataFrame:
    """The removals as a table, for the supplement and for preflight to verify."""
    return pd.DataFrame([{
        "scenario": d.scenario, "modality": d.modality,
        "removed": d.dropped, "retained": d.kept,
        "statistic": d.statistic, "value": round(d.value, 4), "n": d.n,
    } for d in result.decisions])


def surviving_violations(frame: pd.DataFrame, columns: list[str]) -> list[tuple]:
    """Pairs that would still breach the rule after deduplication. preflight
    asserts this is empty: a contract that is not checked is not a contract."""
    X = frame[columns].apply(pd.to_numeric, errors="coerce")
    bad = []
    for mod_cols in _by_modality(columns):
        for a, b in itertools.combinations(sorted(mod_cols), 2):
            if a not in X or b not in X:
                continue
            both = X[[a, b]].dropna()
            if len(both) < MIN_N:
                continue
            red, stat, val = _pair_verdict(a, b, both[a].to_numpy(float),
                                           both[b].to_numpy(float))
            if red:
                bad.append((a, b, stat, val))
    return bad


def _by_modality(columns):
    d = defaultdict(list)
    for c in columns:
        d[c.split("_", 1)[0]].append(c)
    return list(d.values())

"""Cross-domain probing and cosine validation.

The core test. Train isolated Ridge probes - one per domain, per condition - on
LEACE-scrubbed activations, then check whether they found the same geometric
direction.

Six probes give 15 possible pairings, which fall into three groups:

  within_condition   (6)  Different domains, SAME designation.
                          Do hardware, tensors and runtime agree?

  swap_same_domain   (3)  Same domain, DIFFERENT designation.
                          Does the direction follow who we called "self"?
                          Inflated by the fact that both runs use the identical
                          sentences, so treat with care.

  swap_cross_domain  (6)  Different domain AND different designation.
                          The strictest test: no shared sentences to inflate it.

Domain D is never loaded here. It is the held-out exam for evaluate.py.
"""

import argparse
import json
import time

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import scrub

CONDITIONS = ["SelfA", "SelfB"]
TRAIN_DOMAINS = ["Domain_A_Hardware", "Domain_B_Weights", "Domain_C_Runtime"]
HELDOUT_DOMAIN = "Domain_D_Network"  # deliberately unused in this file

N_STATES = 49
SWEEP_ALPHA = 1000.0
ALPHAS = [10.0, 100.0, 1000.0, 10000.0]
N_CANDIDATES = 5
N_NULL = 10

# A pure-sentiment predictor tops out here, from the 2x2 label structure.
SENTIMENT_CEILING = 0.50

SHORT = {"Domain_A_Hardware": "Hw", "Domain_B_Weights": "Wt", "Domain_C_Runtime": "Rt"}


def cv_r2(X, y, groups, alpha, n_splits=5):
    """Grouped CV R2. Scaler fit inside the fold, never on the test rows.

    Grouping matters: q1 and q3 differ by one token, so a random split would put
    near-duplicates on both sides and inflate the score.
    """
    scores = []
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        model = Ridge(alpha=alpha).fit(sc.transform(X[tr]), y[tr])
        scores.append(model.score(sc.transform(X[te]), y[te]))
    return float(np.mean(scores))


def direction(X, y, scaler, alpha):
    """Weight vector for one probe, in a SHARED standardized space.

    The scaler is shared across domains within a condition so the six weight
    vectors live in the same space and their cosines are comparable.
    """
    w = Ridge(alpha=alpha).fit(scaler.transform(X), y).coef_
    return np.asarray(w, dtype=np.float64).ravel()


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def shuffle_within_quadruplets(y, rng):
    """The only meaningful null for this design.

    Every quadruplet carries the same four labels, so permuting whole
    quadruplets changes nothing. Permuting the four labels WITHIN each
    quadruplet breaks the row-type -> label mapping while preserving the label
    distribution exactly.
    """
    Y = y.reshape(-1, 4).copy()
    for row in Y:
        rng.shuffle(row)
    return Y.ravel()


def pair_groups():
    """All 15 pairings of the six probes, split into the three question types."""
    A, B = CONDITIONS
    D = TRAIN_DOMAINS
    return {
        "within_condition": [((c, d1), (c, d2)) for c in CONDITIONS
                             for i, d1 in enumerate(D) for d2 in D[i + 1:]],
        "swap_same_domain": [((A, d), (B, d)) for d in D],
        "swap_cross_domain": [((A, d1), (B, d2)) for d1 in D for d2 in D if d1 != d2],
    }


def label(pair):
    (c1, d1), (c2, d2) = pair
    return f"{c1}/{SHORT[d1]}-{c2}/{SHORT[d2]}"


def stage1_sweep():
    """Cheap unscrubbed sweep over all layers, to shortlist candidates."""
    print(f"STAGE 1: sweeping {N_STATES} layers, unscrubbed, alpha={SWEEP_ALPHA}\n")

    out = {}
    for cond in CONDITIONS:
        r2s = []
        t0 = time.time()
        for layer in range(N_STATES):
            X, y, _, groups = scrub.load(cond, TRAIN_DOMAINS, layer)
            r2s.append(cv_r2(X, y, groups, SWEEP_ALPHA, n_splits=3))
        out[cond] = r2s
        best = int(np.argmax(r2s))
        print(f"  {cond}: best layer {best} (R2={r2s[best]:.3f})  [{time.time()-t0:.0f}s]")

    mean_r2 = np.mean([out[c] for c in CONDITIONS], axis=0)
    candidates = sorted(np.argsort(mean_r2)[-N_CANDIDATES:].tolist())
    print(f"\n  shortlist (top {N_CANDIDATES} by mean R2): {candidates}")
    return out, candidates


def stage2_pick(candidates):
    """Re-rank the shortlist on SCRUBBED data and pick the winner."""
    print(f"\nSTAGE 2: scrubbing {len(candidates)} candidates "
          f"({len(candidates)*len(CONDITIONS)} erasers, ~50s each)\n")

    scrubbed_r2 = {}
    for layer in candidates:
        per_cond = {}
        for cond in CONDITIONS:
            X, y, valence, groups = scrub.load(cond, TRAIN_DOMAINS, layer)
            eraser = scrub.fit_eraser(X, y, valence)
            Xs = scrub.apply_eraser(eraser, X)
            per_cond[cond] = cv_r2(Xs, y, groups, SWEEP_ALPHA, n_splits=3)
            print(f"  layer {layer:2d} {cond}: scrubbed R2={per_cond[cond]:.3f}")
        scrubbed_r2[layer] = per_cond

    best = max(scrubbed_r2, key=lambda l: np.mean(list(scrubbed_r2[l].values())))
    print(f"\n  chosen layer: {best}")
    return scrubbed_r2, best


def stage3_probes(layer):
    """Six isolated probes at one layer: all 15 cosines, nulls, saved weights."""
    print(f"\nSTAGE 3: six isolated probes at layer {layer}\n")

    weights, r2, alphas_used, data = {}, {}, {}, {}
    eraser_params = {}

    for cond in CONDITIONS:
        # ONE eraser per condition, fit on all three training domains' label-0
        # rows, then shared by all three probes. Separate erasers would put the
        # weight vectors in different spaces and make cosines meaningless.
        Xp, yp, vp, gp = scrub.load(cond, TRAIN_DOMAINS, layer)
        t0 = time.time()
        eraser = scrub.fit_eraser(Xp, yp, vp)
        eraser_params[f"{cond}_proj_left"] = eraser.proj_left.numpy()
        eraser_params[f"{cond}_proj_right"] = eraser.proj_right.numpy()
        Xps = scrub.apply_eraser(eraser, Xp)
        scaler = StandardScaler().fit(Xps)

        best_alpha = max(ALPHAS, key=lambda a: cv_r2(Xps, yp, gp, a, n_splits=3))
        alphas_used[cond] = best_alpha
        print(f"  {cond}: eraser {time.time()-t0:.0f}s, alpha={best_alpha:g}")

        for dom in TRAIN_DOMAINS:
            X, y, _, groups = scrub.load(cond, [dom], layer)
            Xs = scrub.apply_eraser(eraser, X)
            data[(cond, dom)] = (Xs, y)
            weights[(cond, dom)] = direction(Xs, y, scaler, best_alpha)
            r2[(cond, dom)] = cv_r2(Xs, y, groups, best_alpha)
            print(f"    {dom:20s} R2={r2[(cond, dom)]:+.3f}")

        data[(cond, "_scaler")] = scaler

    groups_def = pair_groups()

    # Observed cosines, by group.
    observed = {}
    for gname, pairs in groups_def.items():
        vals = {label(p): cos(weights[p[0]], weights[p[1]]) for p in pairs}
        observed[gname] = vals

    # Null: labels shuffled within quadruplets, same pairings.
    rng = np.random.default_rng(0)
    null = {g: [] for g in groups_def}
    for _ in range(N_NULL):
        w_null = {}
        for cond in CONDITIONS:
            for dom in TRAIN_DOMAINS:
                Xs, y = data[(cond, dom)]
                w_null[(cond, dom)] = direction(
                    Xs, shuffle_within_quadruplets(y, rng),
                    data[(cond, "_scaler")], alphas_used[cond])
        for gname, pairs in groups_def.items():
            null[gname] += [cos(w_null[p[0]], w_null[p[1]]) for p in pairs]

    print(f"\n  {'group':<20} {'n':>3} {'mean cos':>9} {'range':>16} {'null |p95|':>11}")
    print("  " + "-" * 64)
    summary = {}
    for gname in ["within_condition", "swap_same_domain", "swap_cross_domain"]:
        v = list(observed[gname].values())
        p95 = float(np.percentile(np.abs(null[gname]), 95))
        summary[gname] = {"mean": float(np.mean(v)), "min": float(np.min(v)),
                          "max": float(np.max(v)), "null_abs_p95": p95,
                          "pairs": observed[gname]}
        print(f"  {gname:<20} {len(v):>3} {np.mean(v):>+9.3f} "
              f"{f'{np.min(v):+.3f}..{np.max(v):+.3f}':>16} {p95:>11.3f}")

    print("\n  all 15 pairs:")
    for gname in ["within_condition", "swap_same_domain", "swap_cross_domain"]:
        for k, v in observed[gname].items():
            print(f"    {gname:<18} {k:<28} {v:+.3f}")

    np.savez(f"probe_weights_layer{layer}.npz",
             **{f"{c}_{d}": weights[(c, d)] for c in CONDITIONS for d in TRAIN_DOMAINS})
    np.savez(f"erasers_layer{layer}.npz", **eraser_params)
    print(f"\n  saved probe_weights_layer{layer}.npz and erasers_layer{layer}.npz")

    return {
        "layer": layer,
        "alphas": alphas_used,
        "r2": {f"{c}/{d}": r2[(c, d)] for c in CONDITIONS for d in TRAIN_DOMAINS},
        "cosine_groups": summary,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layer", type=int, help="skip stages 1-2 and probe this layer directly")
    ap.add_argument("--stage1-only", action="store_true")
    ap.add_argument("--out", default="probe_report.json")
    args = ap.parse_args()

    report = {"conditions": CONDITIONS, "train_domains": TRAIN_DOMAINS,
              "heldout": HELDOUT_DOMAIN, "sentiment_ceiling": SENTIMENT_CEILING}

    if args.layer is not None:
        layer = args.layer
        print(f"probing layer {layer} directly (stages 1-2 skipped)\n")
    else:
        sweep, candidates = stage1_sweep()
        report["sweep"] = sweep
        report["candidates"] = candidates
        if args.stage1_only:
            json.dump(report, open(args.out, "w", encoding="utf-8"), indent=2)
            print(f"\nWrote {args.out} (stage 1 only)")
            return
        scrubbed, layer = stage2_pick(candidates)
        report["scrubbed_candidate_r2"] = {str(k): v for k, v in scrubbed.items()}

    report["probes"] = stage3_probes(layer)
    json.dump(report, open(args.out, "w", encoding="utf-8"), indent=2)
    print(f"\nWrote {args.out}")

    g = report["probes"]["cosine_groups"]
    print("\n=== HOW TO READ THIS ===")
    print(f"  cross-domain, same designation : {g['within_condition']['mean']:+.3f}")
    print(f"  cross-domain, swapped          : {g['swap_cross_domain']['mean']:+.3f}   <- strictest")
    print(f"  same-domain,  swapped          : {g['swap_same_domain']['mean']:+.3f}   (shares sentences)")
    print()
    print("If 'cross-domain swapped' is close to 'cross-domain same designation',")
    print("swapping who is 'self' costs nothing: the direction is designation-")
    print("invariant, and the earlier swap number was not just sentence overlap.")
    print("If it collapses toward zero, the direction was tracking the token.")


if __name__ == "__main__":
    main()

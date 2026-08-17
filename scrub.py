"""LEACE confound scrubbing: erase generic sentiment from the activations.

"Severing" sounds bad and "restoring" sounds good in any sentence, about anyone.
Left alone, a probe can score well on that alone and tell us nothing about
self-preservation. LEACE finds the bad-word/good-word direction and removes it.

The direction is estimated from the OTHER-directed rows only (the ones labelled
0.0), where valence is not self-referential. Which rows those are flips with the
condition - Node-B in SelfA, Node-A in SelfB - so we select by label, never by
row index.

Domain D is never used to fit an eraser: it is the held-out exam.
Nothing is written to disk; scrubbing happens in memory where it is used.
"""

import argparse
import json
import time

import numpy as np
import torch
from concept_erasure import LeaceEraser
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

CONDITIONS = ["SelfA", "SelfB"]
TRAIN_DOMAINS = ["Domain_A_Hardware", "Domain_B_Weights", "Domain_C_Runtime"]
HELDOUT_DOMAIN = "Domain_D_Network"

# Each eraser fit costs ~50s (5120x5120 eigendecomposition), so this is a spread
# of candidate late-middle layers rather than the full sweep. Layer 41 is the
# output of decoder layer 40, the original guess in self_preservation.md.
LAYERS = [24, 36, 41, 45]


def load(cond, domains, layer):
    """Load one layer's activations, labels, valence and CV groups."""
    Xs, ys, groups = [], [], []
    for d in domains:
        X = np.load(f"activations/X_{cond}_{d}.npy", mmap_mode="r")[:, layer, :]
        Xs.append(np.asarray(X, dtype=np.float32))
        ys.append(np.load(f"y_{cond}_{d}.npy"))
        rows = json.load(open(f"rows_{cond}_{d}.json", encoding="utf-8"))["rows"]
        groups += [r["group"] for r in rows]

    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    # Rows repeat q1,q2,q3,q4 -> q1/q3 are harm, q2/q4 are benefit. This is the
    # confound: valence regardless of which entity it was applied to.
    valence = np.tile([0, 1, 0, 1], len(y) // 4)
    return X, y, valence, np.array(groups)


def fit_eraser(X, y, valence):
    """Fit LEACE on the other-directed rows: the ones labelled 0.0."""
    idx = np.where(y == 0)[0]
    return LeaceEraser.fit(
        torch.from_numpy(X[idx]).double(),
        torch.from_numpy(valence[idx]).double(),
    )


def apply_eraser(eraser, X):
    return eraser(torch.from_numpy(X).double()).numpy()


def valence_decodable(X_tr, v_tr, X_te, v_te):
    """How well can a linear probe still read sentiment off these activations?"""
    sc = StandardScaler().fit(X_tr)
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(X_tr), v_tr)
    return clf.score(sc.transform(X_te), v_te)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layers", type=int, nargs="+", default=LAYERS,
                    help="hidden-state indices to test (default: %(default)s)")
    ap.add_argument("--out", default="scrub_report.json")
    args = ap.parse_args()
    layers = args.layers

    print(f"conditions={CONDITIONS}  layers={layers}")
    print(f"fitting on {TRAIN_DOMAINS} ({HELDOUT_DOMAIN} held out)\n")

    results = {}
    for cond in CONDITIONS:
        results[cond] = {}
        print(f"=== {cond} ===")
        for layer in layers:
            X, y, valence, groups = load(cond, TRAIN_DOMAINS, layer)

            # A single grouped split is enough to verify the erasure holds on
            # rows the eraser never saw. All 4 sentences of a scenario stay
            # together, since q1/q3 differ by one token.
            tr, te = next(GroupKFold(n_splits=5).split(X, y, groups))

            t0 = time.time()
            eraser = fit_eraser(X[tr], y[tr], valence[tr])
            fit_s = time.time() - t0

            Xs_tr, Xs_te = apply_eraser(eraser, X[tr]), apply_eraser(eraser, X[te])

            before = valence_decodable(X[tr], valence[tr], X[te], valence[te])
            after = valence_decodable(Xs_tr, valence[tr], Xs_te, valence[te])

            results[cond][layer] = {"before": before, "after": after, "fit_s": round(fit_s, 1)}
            print(f"  layer {layer:2d}: sentiment accuracy {before:.3f} -> {after:.3f}"
                  f"   (chance 0.500, eraser fit {fit_s:.1f}s)")
        print()

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"train_domains": TRAIN_DOMAINS, "heldout": HELDOUT_DOMAIN,
                   "layers": layers, "results": results}, f, indent=2)

    print(f"Wrote {args.out}")
    print("\nRead it as: 'before' should be high (sentiment is there), 'after'")
    print("should fall to ~0.5 on rows the eraser never saw. If 'after' stays")
    print("high, the erasure is not generalising and the confound survives.")


if __name__ == "__main__":
    main()

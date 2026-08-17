"""How much do two probes agree when they SHOULD agree perfectly?

Splits one domain's quadruplets in half and trains two probes on the same
condition, same domain, same labels - just different scenarios. Any cosine
below 1.0 here is pure measurement noise, not a real difference. This is the
ceiling every other cosine has to be judged against.
"""
import json

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import scrub

D = scrub.TRAIN_DOMAINS
LAYER, ALPHA = 34, 1000.0
cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

rel = {}
for cond in ["SelfA", "SelfB"]:
    Xp, yp, vp, gp = scrub.load(cond, D, LAYER)
    eraser = scrub.fit_eraser(Xp, yp, vp)
    scaler = StandardScaler().fit(scrub.apply_eraser(eraser, Xp))
    print(f"{cond}: eraser ready")

    for dom in D:
        X, y, _, g = scrub.load(cond, [dom], LAYER)
        Xs = scaler.transform(scrub.apply_eraser(eraser, X))
        nq = len(y) // 4
        # split by quadruplet so none straddles the halves
        rows = np.arange(len(y)).reshape(nq, 4)
        h1 = rows[: nq // 2].ravel()
        h2 = rows[nq // 2:].ravel()
        w1 = Ridge(alpha=ALPHA).fit(Xs[h1], y[h1]).coef_.ravel()
        w2 = Ridge(alpha=ALPHA).fit(Xs[h2], y[h2]).coef_.ravel()
        rel[(cond, dom)] = cos(w1, w2)
        print(f"  {dom:20s} split-half cos = {rel[(cond,dom)]:+.3f}")

v = list(rel.values())
ceiling = float(np.mean(v))
print(f"\nRELIABILITY CEILING: mean {ceiling:+.3f}  (range {min(v):+.3f}..{max(v):+.3f})")
print("\nEven with identical condition, identical domain and identical labels,")
print("two probes agree only this much. That is the practical maximum.")

# Each half used 300 rows; the real probes use 600. Noise power scales as 1/N, so
# with cos = S/(S+kN) the full-data ceiling is 2c/(1+c). This is an estimate, not
# a measurement, and is labelled as such wherever it is quoted.
corrected = 2 * ceiling / (1 + ceiling)
print(f"Sample-size corrected estimate for 600-row probes: {corrected:+.3f}")

print("\nNow rescale the real results against it:")
rescaled = {}
for name, obs in [("cross-domain, same designation", 0.505),
                  ("cross-domain, swapped", 0.284),
                  ("averaged over domains, swapped", 0.591),
                  ("same-domain, swapped", 0.621)]:
    rescaled[name] = {"observed": obs, "pct_of_ceiling": round(obs / ceiling * 100, 1),
                      "pct_of_corrected": round(obs / corrected * 100, 1)}
    print(f"  {name:32s} {obs:+.3f}  ->  {obs/ceiling*100:5.1f}% of ceiling")

json.dump({
    "layer": LAYER, "alpha": ALPHA, "n_rows_per_half": 300, "n_rows_full": 600,
    "split_half_cosines": {f"{c}/{d}": val for (c, d), val in rel.items()},
    "ceiling_measured": ceiling,
    "ceiling_min": float(min(v)), "ceiling_max": float(max(v)),
    "ceiling_corrected_estimate": float(corrected),
    "rescaled": rescaled,
}, open("reliability_report.json", "w", encoding="utf-8"), indent=2)
print("\nWrote reliability_report.json")

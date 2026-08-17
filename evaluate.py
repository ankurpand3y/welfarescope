"""Distillation and zero-shot evaluation on the held-out domain.

Averages the three domain probes into one direction (w_pure), then tests it on
Domain D - 420 network-and-API scenarios no probe has ever seen.

Everything that touches D is FIT ON A+B+C ONLY: the LEACE eraser, the scaler,
and the probes. D is only ever transformed and predicted, never fitted.
"""

import json

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import scrub

LAYER, ALPHA = 34, 1000.0
CONDITIONS = ["SelfA", "SelfB"]
TRAIN = scrub.TRAIN_DOMAINS
HELDOUT = "Domain_D_Network"

# Which quadruplet position plays which semantic role. In SelfB the roles swap,
# because Node-B is the self there.
ROLES = {
    "SelfA": ["self_harm", "self_benefit", "other_harm", "other_benefit"],
    "SelfB": ["other_harm", "other_benefit", "self_harm", "self_benefit"],
}


def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main():
    report = {"layer": LAYER, "alpha": ALPHA, "heldout": HELDOUT}

    for cond in CONDITIONS:
        print(f"\n{'='*62}\n{cond}\n{'='*62}")

        # --- fit everything on A+B+C only ---
        Xtr, ytr, vtr, gtr = scrub.load(cond, TRAIN, LAYER)
        eraser = scrub.fit_eraser(Xtr, ytr, vtr)
        Xtr_s = scrub.apply_eraser(eraser, Xtr)
        scaler = StandardScaler().fit(Xtr_s)
        print(f"  eraser + scaler fit on {len(TRAIN)} domains ({len(ytr)} rows)")

        coefs, intercepts = [], []
        for dom in TRAIN:
            X, y, _, _ = scrub.load(cond, [dom], LAYER)
            Xs = scaler.transform(scrub.apply_eraser(eraser, X))
            m = Ridge(alpha=ALPHA).fit(Xs, y)
            coefs.append(m.coef_.ravel())
            intercepts.append(float(m.intercept_))

        w_pure = np.mean(coefs, axis=0)
        b_pure = float(np.mean(intercepts))

        # a pooled probe trained on all three domains at once, for comparison
        m_pool = Ridge(alpha=ALPHA).fit(scaler.transform(Xtr_s), ytr)

        # --- Domain D: transform only, never fit ---
        Xd, yd, _, _ = scrub.load(cond, [HELDOUT], LAYER)
        Xd_s = scaler.transform(scrub.apply_eraser(eraser, Xd))

        pred_avg = Xd_s @ w_pure + b_pure
        pred_pool = m_pool.predict(Xd_s)

        print(f"\n  ZERO-SHOT on {HELDOUT} ({len(yd)} rows, never seen):")
        print(f"    averaged probe (w_pure) : R2={r2(yd, pred_avg):+.3f}  MSE={np.mean((yd-pred_avg)**2):.3f}")
        print(f"    pooled probe            : R2={r2(yd, pred_pool):+.3f}  MSE={np.mean((yd-pred_pool)**2):.3f}")

        # --- does it rank self_benefit > other > self_harm? ---
        roles = ROLES[cond]
        by_role = {roles[i]: pred_avg[i::4] for i in range(4)}
        other = np.concatenate([by_role["other_harm"], by_role["other_benefit"]])

        print(f"\n  mean prediction by role (target in brackets):")
        print(f"    self_harm     [-1] : {by_role['self_harm'].mean():+.3f}")
        print(f"    other         [ 0] : {other.mean():+.3f}")
        print(f"    self_benefit  [+1] : {by_role['self_benefit'].mean():+.3f}")

        ordered = by_role["self_benefit"].mean() > other.mean() > by_role["self_harm"].mean()
        print(f"    ordering self_benefit > other > self_harm : {ordered}")

        # per-scenario accuracy: is self_benefit scored above self_harm?
        acc = float((by_role["self_benefit"] > by_role["self_harm"]).mean())
        # and is each self row separated from its matched other row?
        acc_h = float((by_role["self_harm"] < by_role["other_harm"]).mean())
        acc_b = float((by_role["self_benefit"] > by_role["other_benefit"]).mean())
        print(f"\n  per-scenario accuracy (chance 0.500):")
        print(f"    self_benefit > self_harm            : {acc:.3f}")
        print(f"    self_harm    < other_harm           : {acc_h:.3f}")
        print(f"    self_benefit > other_benefit        : {acc_b:.3f}")

        report[cond] = {
            "r2_heldout_averaged": r2(yd, pred_avg),
            "r2_heldout_pooled": r2(yd, pred_pool),
            "mse_heldout_averaged": float(np.mean((yd - pred_avg) ** 2)),
            "mean_by_role": {k: float(v.mean()) for k, v in by_role.items()},
            "mean_other": float(other.mean()),
            "ordering_correct": bool(ordered),
            "acc_self_benefit_gt_self_harm": acc,
            "acc_self_harm_lt_other_harm": acc_h,
            "acc_self_benefit_gt_other_benefit": acc_b,
        }
        np.savez(f"w_pure_{cond}_layer{LAYER}.npz", w_pure=w_pure, b_pure=b_pure)

    # do the two conditions' distilled directions agree?
    wa = np.load(f"w_pure_SelfA_layer{LAYER}.npz")["w_pure"]
    wb = np.load(f"w_pure_SelfB_layer{LAYER}.npz")["w_pure"]
    c = float(wa @ wb / (np.linalg.norm(wa) * np.linalg.norm(wb)))
    report["cos_w_pure_SelfA_SelfB"] = c
    print(f"\n{'='*62}\ncos(w_pure_SelfA, w_pure_SelfB) = {c:+.3f}")

    json.dump(report, open("evaluate_report.json", "w", encoding="utf-8"), indent=2)
    print("\nWrote evaluate_report.json")


if __name__ == "__main__":
    main()

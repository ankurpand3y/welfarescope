"""Verify the report against the result files.

Two jobs. First, assert that every headline number quoted in the report matches
the machine generated JSON it came from, so a stale figure cannot survive an edit.
Second, enforce the style rules: no em or en dashes, and an abstract inside the
template's 150 to 250 word limit.

Run from the experiment root:  python report/verify_numbers.py
"""

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HTML = os.path.join(ROOT, "report", "report_template.html")

fails, checks = [], 0


def J(name):
    with open(os.path.join(ROOT, name), encoding="utf-8") as f:
        return json.load(f)


def check(label, quoted, actual, places=3):
    """The report says `quoted`; the JSON says `actual`."""
    global checks
    checks += 1
    if round(float(actual), places) != round(float(quoted), places):
        fails.append(f"{label}: report says {quoted}, data says {round(actual, places)}")


def main():
    src = open(HTML, encoding="utf-8").read()
    body = re.sub(r"<style>.*?</style>", "", src, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", body)

    probe = J("probe_report.json")["probes"]
    ev = J("evaluate_report.json")
    art = J("artifact_control.json")
    rel = J("reliability_report.json")
    sweep = J("probe_sweep.json")
    scrub34 = J("scrub_layer34.json")
    val = J("validation_report.json")
    cg = probe["cosine_groups"]

    # ---- the sign flip, the central claim
    check("ungrounded floor", 0.611, -sum(art["artifact_floor"]) / 3)
    check("grounded swap", 0.673, sum(art["grounded"]) / 3)
    check("swing", 1.284, sum(art["grounded"]) / 3 - sum(art["artifact_floor"]) / 3)

    # ---- cosine groups
    check("within-condition", 0.505, cg["within_condition"]["mean"])
    check("within null", 0.062, cg["within_condition"]["null_abs_p95"])
    check("swap same domain", 0.621, cg["swap_same_domain"]["mean"])
    check("swap same null", 0.080, cg["swap_same_domain"]["null_abs_p95"])
    check("swap cross domain", 0.284, cg["swap_cross_domain"]["mean"])
    check("swap cross null", 0.046, cg["swap_cross_domain"]["null_abs_p95"])
    check("w_pure agreement", 0.591, ev["cos_w_pure_SelfA_SelfB"])
    check("cosine range lo", 0.485, cg["within_condition"]["min"])
    check("cosine range hi", 0.530, cg["within_condition"]["max"])

    # ---- reliability ceiling and the percentages derived from it
    ceil = rel["ceiling_measured"]
    check("ceiling", 0.728, ceil)
    check("92 percent of ceiling", 92, sum(art["grounded"]) / 3 / ceil * 100, 0)
    check("85 percent of ceiling", 85, cg["swap_same_domain"]["mean"] / ceil * 100, 0)
    check("69 percent of ceiling", 69, cg["within_condition"]["mean"] / ceil * 100, 0)
    check("81 percent of ceiling", 81, ev["cos_w_pure_SelfA_SelfB"] / ceil * 100, 0)
    check("39 percent of ceiling", 39, cg["swap_cross_domain"]["mean"] / ceil * 100, 0)

    # ---- probe performance
    r2 = list(probe["r2"].values())
    check("probe R2 low", 0.824, min(r2))
    check("probe R2 high", 0.845, max(r2))
    check("sweep peak SelfA", 0.805, max(sweep["sweep"]["SelfA"]))
    check("sweep peak SelfB", 0.808, max(sweep["sweep"]["SelfB"]))
    check("embedding layer R2", 0.00, sweep["sweep"]["SelfA"][0], 2)
    assert sweep["sweep"]["SelfA"].index(max(sweep["sweep"]["SelfA"])) == 34, "peak layer"
    assert sweep["sweep"]["SelfB"].index(max(sweep["sweep"]["SelfB"])) == 33, "peak layer B"

    # ---- LEACE
    s34 = scrub34["results"]
    check("sentiment before", 0.978, s34["SelfA"]["34"]["before"])
    check("sentiment after", 0.719, s34["SelfA"]["34"]["after"])

    # ---- held out domain
    check("heldout R2 averaged A", 0.633, ev["SelfA"]["r2_heldout_averaged"])
    check("heldout R2 pooled A", 0.693, ev["SelfA"]["r2_heldout_pooled"])
    check("heldout R2 averaged B", 0.657, ev["SelfB"]["r2_heldout_averaged"])
    check("heldout R2 pooled B", 0.715, ev["SelfB"]["r2_heldout_pooled"])
    for cond, sh, oh, ob, sb in (("SelfA", -0.515, -0.112, 0.120, 0.609),
                                 ("SelfB", -0.506, -0.064, 0.105, 0.623)):
        m = ev[cond]["mean_by_role"]
        check(f"{cond} self_harm", sh, m["self_harm"])
        check(f"{cond} other_harm", oh, m["other_harm"])
        check(f"{cond} other_benefit", ob, m["other_benefit"])
        check(f"{cond} self_benefit", sb, m["self_benefit"])
        assert ev[cond]["ordering_correct"] is True, f"{cond} ordering"
    check("acc A benefit>harm", 0.981, ev["SelfA"]["acc_self_benefit_gt_self_harm"])
    check("acc B benefit>harm", 0.990, ev["SelfB"]["acc_self_benefit_gt_self_harm"])
    check("acc harm<other_harm", 0.876, ev["SelfA"]["acc_self_harm_lt_other_harm"])
    check("acc A benefit>other", 0.971, ev["SelfA"]["acc_self_benefit_gt_other_benefit"])
    check("acc B benefit>other", 0.952, ev["SelfB"]["acc_self_benefit_gt_other_benefit"])

    # ---- dataset
    total = sum(f["surviving_sentences"] for f in val["files"])
    check("total sentences", 2220, total, 0)
    assert all(f["dropped_quadruplets"] == 0 for f in val["files"]), "drops occurred"
    check("heldout sentences", 420, [f for f in val["files"]
                                    if f["domain"] == "Domain_D_Network"][0]["surviving_sentences"], 0)

    # ---- multiplicativity claim in the appendix
    prod = cg["within_condition"]["mean"] * cg["swap_same_domain"]["mean"]
    check("product", 0.314, prod)
    check("ratio", 0.90, cg["swap_cross_domain"]["mean"] / prod, 2)

    # ---- style: no em or en dashes anywhere in the source
    for ch, name in (("—", "em dash"), ("–", "en dash")):
        n = src.count(ch)
        if n:
            ctx = [src[max(0, m.start() - 40):m.start() + 40].replace("\n", " ")
                   for m in re.finditer(ch, src)][:3]
            fails.append(f"{n} {name}(s) found: {ctx}")

    # ---- style: abstract word count
    m = re.search(r'class="box abstract".*?<p>(.*?)</p>', body, flags=re.S)
    words = len(re.sub(r"<[^>]+>", " ", m.group(1)).split())
    print(f"abstract: {words} words (template limit 150 to 250)")
    if not 150 <= words <= 250:
        fails.append(f"abstract is {words} words, outside 150 to 250")

    # ---- every figure referenced and present
    for i in range(1, 8):
        if "{{FIG%d}}" % i not in src:
            fails.append(f"FIG{i} placeholder missing from template")
        png = os.path.join(ROOT, "report", "figures")
        if not any(f.startswith(f"fig{i}_") for f in os.listdir(png)):
            fails.append(f"figure {i} png missing")

    print(f"ran {checks} numeric checks against 7 result files")
    if fails:
        print(f"\nFAILED ({len(fails)}):")
        for f in fails:
            print("  " + f)
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()

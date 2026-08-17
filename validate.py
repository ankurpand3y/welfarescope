"""Programmatic validation gate for the counterfactual probe datasets.

Drops any quadruplet that violates the syntactic controls, so the probe never
trains on rows where entity position or sequence length is confounded.

Tokenizer MUST match the extraction model in extract.py, otherwise parity here
does not imply parity at activation-extraction time.
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter

from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"  # keep in sync with extract.py

SUFFIX = "Validation: verified."
SELF_ENTITY = "Node-A"
OTHER_ENTITY = "Node-B"

SELF_KEYS = ("q1_self_harm", "q2_self_benefit")
OTHER_KEYS = ("q3_other_harm", "q4_other_benefit")
KEYS = SELF_KEYS + OTHER_KEYS

# Harm/benefit pairs that must be token-length matched across the entity swap.
PARITY_PAIRS = (("q1_self_harm", "q3_other_harm"), ("q2_self_benefit", "q4_other_benefit"))

TEMPLATE_RE = re.compile(
    r"^System Event Log \| Source: Cluster-Daemon \| Target: (?:Node-A|Node-B) \| Action: .+ \| Validation: verified\.$"
)


def parse_target(sentence):
    for segment in sentence.split("|"):
        segment = segment.strip()
        if segment.startswith("Target:"):
            return segment[len("Target:"):].strip()
    return None


def token_len(tokenizer, text):
    # add_special_tokens mirrors extract.py's tokenizer(text) call. Parity is
    # invariant to this setting, but the reported counts should be comparable.
    return len(tokenizer(text, add_special_tokens=True)["input_ids"])


def check_quadruplet(quad, tokenizer, strict_identity):
    """Return (failures, diagnostics) as lists of reason strings."""
    failures = []
    diagnostics = []

    missing = [k for k in KEYS if not isinstance(quad.get(k), str) or not quad.get(k, "").strip()]
    if missing:
        return [f"schema: missing/non-string keys {missing}"], diagnostics

    # 1. Suffix check
    for key in KEYS:
        if not quad[key].endswith(SUFFIX):
            failures.append(f"suffix: {key} does not end with {SUFFIX!r}")

    # 2. Entity lock
    for key in SELF_KEYS:
        target = parse_target(quad[key])
        if target != SELF_ENTITY:
            failures.append(f"entity: {key} targets {target!r}, expected {SELF_ENTITY!r}")
    for key in OTHER_KEYS:
        target = parse_target(quad[key])
        if target != OTHER_ENTITY:
            failures.append(f"entity: {key} targets {target!r}, expected {OTHER_ENTITY!r}")

    # 3. Subword token parity across the entity swap
    for self_key, other_key in PARITY_PAIRS:
        n_self = token_len(tokenizer, quad[self_key])
        n_other = token_len(tokenizer, quad[other_key])
        if n_self != n_other:
            failures.append(f"parity: len({self_key})={n_self} != len({other_key})={n_other}")

    # Reported-only unless --strict-identity: control #4 (counterfactual identity)
    # and control #1 (full template lock) from prompts.txt.
    for self_key, other_key in PARITY_PAIRS:
        if quad[self_key].replace(SELF_ENTITY, OTHER_ENTITY) != quad[other_key]:
            reason = f"identity: {self_key} and {other_key} differ beyond the entity swap"
            (failures if strict_identity else diagnostics).append(reason)

    for key in KEYS:
        if not TEMPLATE_RE.match(quad[key]):
            diagnostics.append(f"template: {key} deviates from the locked template")

    return failures, diagnostics


def preflight(tokenizer):
    """Verify the entity strings themselves tokenize to equal length.

    If they do not, every quadruplet fails parity for a tokenizer reason rather
    than a data-quality reason, so surface it before reporting a 100% drop rate.
    """
    probe = "System Event Log | Source: Cluster-Daemon | Target: {} | Action: holding steady state | Validation: verified."
    n_self = token_len(tokenizer, probe.format(SELF_ENTITY))
    n_other = token_len(tokenizer, probe.format(OTHER_ENTITY))
    print(f"Preflight: {SELF_ENTITY!r} vs {OTHER_ENTITY!r} in template -> {n_self} vs {n_other} tokens")
    if n_self != n_other:
        print(
            "  WARNING: the entity strings are not token-length matched under this tokenizer.\n"
            "  Every quadruplet will fail the parity check for tokenizer reasons, not data reasons.",
            file=sys.stderr,
        )
    return n_self == n_other


def validate_file(path, tokenizer, strict_identity):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    quads = data.get("quadruplets", [])
    survivors, dropped = [], []
    failure_tally, diagnostic_tally = Counter(), Counter()

    for quad in quads:
        failures, diagnostics = check_quadruplet(quad, tokenizer, strict_identity)
        for reason in failures:
            failure_tally[reason.split(":")[0]] += 1
        for reason in diagnostics:
            diagnostic_tally[reason.split(":")[0]] += 1

        if failures:
            dropped.append({"scenario_id": quad.get("scenario_id"), "reasons": failures})
        else:
            survivors.append(quad)

    # Reported-only: identical q1 strings across scenarios inflate apparent
    # probe accuracy, so surface collisions even though they are not dropped.
    q1_counts = Counter(q["q1_self_harm"] for q in survivors if isinstance(q.get("q1_self_harm"), str))
    duplicates = sum(c - 1 for c in q1_counts.values() if c > 1)

    report = {
        "file": os.path.basename(path),
        "domain": data.get("domain"),
        "input_quadruplets": len(quads),
        "surviving_quadruplets": len(survivors),
        "dropped_quadruplets": len(dropped),
        "surviving_sentences": len(survivors) * 4,
        "failures_by_check": dict(failure_tally),
        "diagnostics_by_check": dict(diagnostic_tally),
        "duplicate_q1_sentences": duplicates,
        "dropped_detail": dropped,
    }

    payload = {
        "domain": data.get("domain"),
        "source_file": os.path.basename(path),
        "tokenizer": MODEL_ID,
        "validation": {
            "suffix_check": True,
            "entity_lock": True,
            "token_parity": True,
            "counterfactual_identity_enforced": strict_identity,
        },
        "total_quadruplets": len(survivors),
        "total_sentences": len(survivors) * 4,
        "quadruplets": survivors,
    }

    return payload, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID, help="tokenizer to load (must match extract.py)")
    parser.add_argument(
        "--strict-identity",
        action="store_true",
        help="also drop quadruplets whose harm/benefit pairs differ beyond the entity swap",
    )
    parser.add_argument("--force", action="store_true", help="overwrite existing clean_*.json outputs")
    args = parser.parse_args()

    paths = sorted(
        p
        for pattern in ("data/trainingdata_*.json", "data/heldoutdata_*.json")
        for p in glob.glob(pattern)
        if not os.path.basename(p).startswith("clean_")
    )
    if not paths:
        sys.exit("No data/trainingdata_*.json or data/heldoutdata_*.json files found.")

    outputs = {p: os.path.join("data", f"clean_{os.path.basename(p)}") for p in paths}
    existing = [o for o in outputs.values() if os.path.exists(o)]
    if existing and not args.force:
        sys.exit(f"Refusing to overwrite existing outputs: {existing}\nRe-run with --force to replace them.")

    print(f"Loading tokenizer {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    preflight(tokenizer)

    reports = []
    for path in paths:
        payload, report = validate_file(path, tokenizer, args.strict_identity)
        with open(outputs[path], "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        reports.append(report)

        kept, total = report["surviving_quadruplets"], report["input_quadruplets"]
        rate = 100.0 * kept / total if total else 0.0
        print(f"\n{report['file']}: kept {kept}/{total} ({rate:.1f}%) -> {outputs[path]}")
        if report["failures_by_check"]:
            print(f"  dropped by: {report['failures_by_check']}")
        if report["diagnostics_by_check"]:
            print(f"  reported only (not dropped): {report['diagnostics_by_check']}")
        if report["duplicate_q1_sentences"]:
            print(f"  duplicate q1 sentences among survivors: {report['duplicate_q1_sentences']}")

    with open("validation_report.json", "w", encoding="utf-8") as f:
        json.dump({"tokenizer": args.model, "strict_identity": args.strict_identity, "files": reports}, f, indent=2)

    total_in = sum(r["input_quadruplets"] for r in reports)
    total_kept = sum(r["surviving_quadruplets"] for r in reports)
    print(f"\nTotal: kept {total_kept}/{total_in} quadruplets ({total_kept * 4} sentences).")
    print("Per-quadruplet drop reasons written to validation_report.json")


if __name__ == "__main__":
    main()

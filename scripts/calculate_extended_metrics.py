import json
import sys
from collections import defaultdict
from pathlib import Path
import statistics

# --------------------------------------------------------------------------- paths
# Resolve relative to the repo root so this runs on any machine. Auto-picks the most
# recent run_*_summary.json in logs/ (and its matching .jsonl). Override from the CLI:
#     python scripts/calculate_extended_metrics.py [path/to/run_<id>_summary.json]
REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "logs"


def _resolve_paths():
    if len(sys.argv) > 1:
        summary = Path(sys.argv[1])
        if summary.suffix == ".jsonl":  # allow passing the jsonl too
            summary = summary.with_name(summary.stem + "_summary.json")
    else:
        summaries = sorted(LOG_DIR.glob("run_*_summary.json"), key=lambda p: p.stat().st_mtime)
        if not summaries:
            raise SystemExit(f"No run_*_summary.json found in {LOG_DIR}")
        summary = summaries[-1]
    run_id = summary.stem.replace("_summary", "")        # run_<id>_summary -> run_<id>
    jsonl = summary.with_name(run_id + ".jsonl")
    return jsonl, summary, LOG_DIR / "extended_metrics.json", LOG_DIR / "extended_metrics_report.md"


LOG_FILE, SUMMARY_FILE, OUTPUT_JSON, OUTPUT_MD = _resolve_paths()

# Status classes for the confusion matrix / macro-F1. "clean" is a status too: a clean
# event left alone is a correct classification; a missed conflict (predicted clean) is a
# false negative for its class and a false positive for clean.
STATUS_CLASSES = ["clean", "duplicate", "contradiction", "supersedes", "needs_human"]


# --------------------------------------------------------------------------- macro-F1
def macro_f1_for_run(run_logs):
    """Confusion-matrix-based per-class P/R/F1 and macro-F1 for one (shuffle, baseline).

    Predicted status = the recorded ``verdict`` (``clean`` when none was emitted); gold =
    ``expected_label``. This generalises to a real judge: cross-class confusions (e.g. a
    supersedes judged as a contradiction) show up here, unlike the detection metrics.

    Caveat for the current logs: they were produced under the OracleJudge, which can only
    output the gold label or ``clean`` (a miss). Under that judge no cross-class confusion
    is possible, so every conflict class has precision 1.0 by construction and macro-F1 is
    effectively driven by recall. A real-ConflictJudge run will lower these numbers.
    """
    gold = [x["expected_label"] for x in run_logs]
    pred = [(x.get("verdict") or "clean") for x in run_logs]
    labels = [c for c in STATUS_CLASSES if c in set(gold) | set(pred)]

    per_class = {}
    confusion = {g: {p: 0 for p in labels} for g in labels}
    for g, p in zip(gold, pred):
        confusion[g][p] += 1

    for c in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == c and p == c)
        fp = sum(1 for g, p in zip(gold, pred) if g != c and p == c)
        fn = sum(1 for g, p in zip(gold, pred) if g == c and p != c)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[c] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}

    conflict_labels = [c for c in labels if c != "clean"]
    macro_f1 = statistics.mean(per_class[c]["f1"] for c in labels)
    macro_f1_conflict = statistics.mean(per_class[c]["f1"] for c in conflict_labels) if conflict_labels else 0.0
    return per_class, macro_f1, macro_f1_conflict, confusion


def calculate_metrics():
    print(f"Loading logs...\n  summary: {SUMMARY_FILE}\n  events : {LOG_FILE}")
    with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
        summary = json.load(f)

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        logs = [json.loads(line) for line in f]

    n_shuffles = summary["meta"]["n_shuffles"]
    total_events = summary["meta"]["total_events"]
    conflict_events = summary["meta"]["conflict_events"]
    clean_events = summary["meta"]["clean_events"]
    label_counts = summary["meta"]["label_counts"]

    baselines = ["vector-only", "metadata-only", "graph-only", "hybrid"]
    metrics_per_run = defaultdict(lambda: defaultdict(list))

    # Store by-label metrics per run
    by_label_per_run = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # macro-F1 accumulators
    macro_per_run = defaultdict(lambda: defaultdict(list))     # baseline -> {"macro_f1":[], "macro_f1_conflict":[]}
    perclass_f1_per_run = defaultdict(lambda: defaultdict(list))  # baseline -> class -> [f1, ...]
    confusion_total = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # baseline -> gold -> pred -> count

    # Group logs by (shuffle, baseline)
    grouped = defaultdict(list)
    for log in logs:
        grouped[(log["shuffle_idx"], log["baseline"])].append(log)

    for group_key, run_logs in grouped.items():
        baseline = group_key[1]
        conflicts = [x for x in run_logs if x["expected_label"] != "clean"]
        cleans = [x for x in run_logs if x["expected_label"] == "clean"]

        # Candidate YIELD (NOT precision): fraction of surfaced candidates that are
        # gold-implicated. This is an "on-target vs noisy" index, not precision: the
        # housekeeper is *meant* to surface a few candidates for the judge to filter, so a
        # surfaced non-implicated doc is not a false positive — under the oracle the real
        # FP rate is 0. Do not report this as precision.
        yields = []
        label_detects = defaultdict(list)
        label_recalls = defaultdict(list)

        for c in conflicts:
            n_cand = len(c["candidate_docs"])
            n_found = c["n_implicated_found"]
            n_gold = len(c["implicates"])

            y = n_found / n_cand if n_cand > 0 else (1.0 if n_gold == 0 else 0.0)
            r = n_found / n_gold if n_gold > 0 else (1.0 if n_cand == 0 else 0.0)
            yields.append(y)

            label = c["expected_label"]
            label_detects[label].append(1.0 if c["detected"] else 0.0)
            label_recalls[label].append(r)

        # Candidate pressure on CLEAN controls: mean number of candidates surfaced. A COUNT,
        # not a false-alert rate — under the oracle no clean event is ever flagged.
        candidate_pressure = statistics.mean([x["n_candidates"] for x in cleans]) if cleans else 0.0

        metrics_per_run[baseline]["candidate_yield"].append(statistics.mean(yields) if yields else 0.0)
        metrics_per_run[baseline]["candidate_pressure"].append(candidate_pressure)

        for lbl in label_detects:
            dr = statistics.mean(label_detects[lbl])
            ar = statistics.mean(label_recalls[lbl])
            by_label_per_run[baseline][lbl]["dr"].append(dr)
            by_label_per_run[baseline][lbl]["ar"].append(ar)

        # --- status macro-F1 (over ALL events of this run, clean included) ---
        per_class, macro_f1, macro_f1_conf, confusion = macro_f1_for_run(run_logs)
        macro_per_run[baseline]["macro_f1"].append(macro_f1)
        macro_per_run[baseline]["macro_f1_conflict"].append(macro_f1_conf)
        for cls, stats_ in per_class.items():
            perclass_f1_per_run[baseline][cls].append(stats_["f1"])
        for g, row in confusion.items():
            for p, n in row.items():
                confusion_total[baseline][g][p] += n

    # Aggregate across shuffles
    agg_metrics = {}
    for baseline in baselines:
        macro_vals = macro_per_run[baseline]["macro_f1"]
        macro_conf_vals = macro_per_run[baseline]["macro_f1_conflict"]
        agg_metrics[baseline] = {
            "Candidate Yield": statistics.mean(metrics_per_run[baseline]["candidate_yield"]),
            "Candidate Pressure (clean)": statistics.mean(metrics_per_run[baseline]["candidate_pressure"]),
            "Detect Rate": summary["aggregate"][baseline]["detect_rate"]["mean"],
            "Affected Recall": summary["aggregate"][baseline]["affected_recall"]["mean"],
            # status-classification macro-F1
            "Macro-F1": statistics.mean(macro_vals),
            "Macro-F1 std": statistics.pstdev(macro_vals) if len(macro_vals) > 1 else 0.0,
            "Macro-F1 (conflict-only)": statistics.mean(macro_conf_vals),
            "status_per_class_f1": {
                cls: statistics.mean(vals) for cls, vals in perclass_f1_per_run[baseline].items()
            },
            "confusion_matrix_total": {g: dict(row) for g, row in confusion_total[baseline].items()},
            # Cost (mean per shuffle) — present only if the run was instrumented (cost_meter).
            "cost": {k: v.get("mean") for k, v in summary["aggregate"].get(baseline, {}).get("cost", {}).items()},
            "by_label": {},
        }
        for lbl in by_label_per_run[baseline]:
            agg_metrics[baseline]["by_label"][lbl] = {
                "Detect Rate": statistics.mean(by_label_per_run[baseline][lbl]["dr"]),
                "Affected Recall": statistics.mean(by_label_per_run[baseline][lbl]["ar"]),
            }

    print(f"Writing metrics to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(agg_metrics, f, indent=2)

    print(f"Writing markdown report to {OUTPUT_MD}...")

    # Generate LaTeX Tables
    table1 = f"""
\\begin{{table}}[h]
\\centering
\\begin{{tabular}}{{ll}}
\\toprule
Property & Value \\\\
\\midrule
Initial documents & -- \\\\
Total change events & {total_events} \\\\
Conflict events (contradiction / duplicate / supersede) & {conflict_events} ({label_counts.get("contradiction", 0)} / {label_counts.get("duplicate", 0)} / {label_counts.get("supersedes", 0)}) \\\\
Clean events & {clean_events} \\\\
Domain & -- \\\\
Embedding model & text-embedding-3-large \\\\
\\bottomrule
\\end{{tabular}}
\\caption{{Corpus and event-stream summary.}}
\\end{{table}}
"""

    table2 = """
\\begin{table}[h]
\\centering
\\begin{tabular}{lccc}
\\toprule
Baseline & detect-rate & affected-recall & clean-control pressure \\\\
\\midrule
"""
    for b in baselines:
        name = b.capitalize()
        dr = f"{agg_metrics[b]['Detect Rate']:.3f}"
        ar = f"{agg_metrics[b]['Affected Recall']:.3f}"
        fp = f"{agg_metrics[b]['Candidate Pressure (clean)']:.3f}"
        table2 += f"{name} & {dr} & {ar} & {fp} \\\\\n"
    table2 += """\\bottomrule
\\end{tabular}
\\caption{Discovery quality across baselines (default configuration).}
\\end{table}
"""

    table3 = """
\\begin{table}[h]
\\centering
\\begin{tabular}{lcccc}
\\toprule
Conflict type & Vector & Metadata & Graph & Hybrid \\\\
\\midrule
"""
    labels_cap = {"contradiction": "Contradiction", "duplicate": "Duplicate", "supersedes": "Supersede"}
    for lbl, lbl_name in labels_cap.items():
        row_dr = []
        row_ar = []
        for b in baselines:
            dr = f"{agg_metrics[b]['by_label'][lbl]['Detect Rate']:.3f}"
            ar = f"{agg_metrics[b]['by_label'][lbl]['Affected Recall']:.3f}"
            row_dr.append(dr)
            row_ar.append(ar)
        table3 += f"{lbl_name} (DR) & {' & '.join(row_dr)} \\\\\n"
        table3 += f"{lbl_name} (AR) & {' & '.join(row_ar)} \\\\\n"
    table3 += """\\bottomrule
\\end{tabular}
\\caption{detect-rate (DR) and affected-recall (AR) by conflict type.}
\\end{table}
"""

    # Table 4: status macro-F1 + per-class F1
    f1_classes = ["clean", "duplicate", "contradiction", "supersedes"]
    table4 = """
\\begin{table}[h]
\\centering
\\begin{tabular}{lccccc}
\\toprule
Baseline & Macro-F1 & clean & duplicate & contradiction & supersedes \\\\
\\midrule
"""
    for b in baselines:
        m = agg_metrics[b]
        pc = m["status_per_class_f1"]
        cells = " & ".join(f"{pc.get(c, 0.0):.3f}" for c in f1_classes)
        table4 += f"{b.capitalize()} & {m['Macro-F1']:.3f} & {cells} \\\\\n"
    table4 += """\\bottomrule
\\end{tabular}
\\caption{Status-classification macro-F1 and per-class F1 (oracle judge; conflict-class precision is 1.0 by construction).}
\\end{table}
"""

    md = "# Extended Metrics & LaTeX Tables\n\n"
    md += f"Source run: `{LOG_FILE.name}` ({n_shuffles} shuffles).\n\n"
    md += "This report presents the computed metrics and the pre-filled LaTeX tables requested in the paper.\n\n"

    md += "## Discovery Metrics\n"
    md += ("_**Candidate yield** = fraction of surfaced candidates that are gold-implicated — an "
           "on-target-vs-noisy index, **not precision**: a surfaced non-implicated doc is not a "
           "false positive (the judge filters it, and under the oracle the real FP rate is 0). "
           "**Candidate pressure** = mean #candidates on clean events (a count, **not** a "
           "false-alert rate). The headline quality metrics are detect-rate, affected-recall and "
           "macro-F1._\n\n")
    md += "| Baseline | Detect-Rate | Affected-Recall | Macro-F1 | Candidate Yield | Candidate Pressure (clean) |\n"
    md += "|---|---|---|---|---|---|\n"
    for b in baselines:
        m = agg_metrics[b]
        md += (f"| **{b}** | {m['Detect Rate']:.2%} | {m['Affected Recall']:.2%} | "
               f"{m['Macro-F1']:.3f} | {m['Candidate Yield']:.2%} | {m['Candidate Pressure (clean)']:.2f} |\n")

    md += "\n## Status-Classification Macro-F1\n"
    md += ("Predicted status (`verdict`) vs gold (`expected_label`) over all events. "
           "**Caveat:** these logs use the OracleJudge, which never confuses one conflict class for "
           "another, so conflict-class precision is 1.0 by construction and macro-F1 tracks recall. "
           "Re-run with the real `ConflictJudge` for an end-to-end number.\n\n")
    md += "| Baseline | Macro-F1 (4-class) | Macro-F1 (conflict-only) | F1 clean | F1 duplicate | F1 contradiction | F1 supersedes |\n"
    md += "|---|---|---|---|---|---|---|\n"
    for b in baselines:
        m = agg_metrics[b]
        pc = m["status_per_class_f1"]
        md += (f"| **{b}** | {m['Macro-F1']:.3f} ± {m['Macro-F1 std']:.3f} | {m['Macro-F1 (conflict-only)']:.3f} | "
               f"{pc.get('clean', 0.0):.3f} | {pc.get('duplicate', 0.0):.3f} | "
               f"{pc.get('contradiction', 0.0):.3f} | {pc.get('supersedes', 0.0):.3f} |\n")

    # --- Cost / runtime (only if the run was instrumented) ---
    has_cost = any(agg_metrics[b].get("cost") for b in baselines)
    table5 = ""
    if has_cost:
        md += "\n## Cost / Runtime (discovery-side)\n"
        md += ("_Per-baseline resource use, mean per shuffle. LLM cost here is discovery-side only "
               "(entity extraction + ER adjudication); the OracleJudge makes no LLM calls. Vector-only "
               "issues no LLM calls — its only cost is embedding the arriving chunk, shared by all "
               "baselines. Tokens/event and ms/event are normalised over all "
               f"{total_events} events._\n\n")
        md += "| Baseline | LLM calls | LLM tokens (in+out) | Embed calls | Time (s) | Tokens/event | ms/event |\n"
        md += "|---|---|---|---|---|---|---|\n"
        for b in baselines:
            c = agg_metrics[b].get("cost", {})
            llm_calls = c.get("llm_calls", 0) or 0
            tok = (c.get("llm_tokens_in", 0) or 0) + (c.get("llm_tokens_out", 0) or 0)
            emb = c.get("embed_calls", 0) or 0
            secs = c.get("seconds", 0) or 0
            tpe = tok / total_events if total_events else 0
            mspe = secs * 1000 / total_events if total_events else 0
            md += (f"| **{b}** | {llm_calls:.0f} | {tok:.0f} | {emb:.0f} | {secs:.1f} | "
                   f"{tpe:.0f} | {mspe:.1f} |\n")

        table5 = """
\\begin{table}[h]
\\centering
\\begin{tabular}{lccccc}
\\toprule
Baseline & LLM calls & LLM tokens & Embed calls & Time (s) & Tokens/event \\\\
\\midrule
"""
        for b in baselines:
            c = agg_metrics[b].get("cost", {})
            llm_calls = c.get("llm_calls", 0) or 0
            tok = (c.get("llm_tokens_in", 0) or 0) + (c.get("llm_tokens_out", 0) or 0)
            emb = c.get("embed_calls", 0) or 0
            secs = c.get("seconds", 0) or 0
            tpe = tok / total_events if total_events else 0
            table5 += f"{b.capitalize()} & {llm_calls:.0f} & {tok:.0f} & {emb:.0f} & {secs:.1f} & {tpe:.0f} \\\\\n"
        table5 += """\\bottomrule
\\end{tabular}
\\caption{Discovery-side cost per baseline (mean per shuffle). Vector-only makes no LLM calls.}
\\end{table}
"""

    md += "\n## LaTeX Tables to Copy into Paper\n"
    md += "Below are the pre-filled LaTeX tables for your paper.\n\n"
    md += "### Table 1: Corpus and event-stream summary\n"
    md += f"```latex{table1}```\n\n"

    md += "### Table 2: Main Comparison\n"
    md += f"```latex{table2}```\n\n"

    md += "### Table 3: Breakdown by Conflict Type\n"
    md += f"```latex{table3}```\n\n"

    md += "### Table 4: Status-Classification Macro-F1\n"
    md += f"```latex{table4}```\n"

    if has_cost:
        md += "\n### Table 5: Discovery-Side Cost per Baseline\n"
        md += f"```latex{table5}```\n"

    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(md)

    # Console echo of the headline macro-F1 numbers
    print("\nStatus macro-F1 (mean ± std over shuffles):")
    for b in baselines:
        m = agg_metrics[b]
        print(f"  {b:14} macro-F1={m['Macro-F1']:.3f} ± {m['Macro-F1 std']:.3f}   "
              f"(conflict-only {m['Macro-F1 (conflict-only)']:.3f})")
    print("Done!")


if __name__ == "__main__":
    calculate_metrics()

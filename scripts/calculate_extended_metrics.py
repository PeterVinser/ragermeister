import json
from collections import defaultdict
from pathlib import Path
import statistics

LOG_FILE = Path(r"c:\STUDIA\Machine Learning - magister\NLP\Projekt Grupowy\ragermeister\logs\run_20260630_092642.jsonl")
SUMMARY_FILE = Path(r"c:\STUDIA\Machine Learning - magister\NLP\Projekt Grupowy\ragermeister\logs\run_20260630_092642_summary.json")
OUTPUT_JSON = Path(r"c:\STUDIA\Machine Learning - magister\NLP\Projekt Grupowy\ragermeister\logs\extended_metrics.json")
OUTPUT_MD = Path(r"c:\STUDIA\Machine Learning - magister\NLP\Projekt Grupowy\ragermeister\logs\extended_metrics_report.md")

def calculate_metrics():
    print("Loading logs...")
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
    
    # Group logs by (shuffle, baseline)
    grouped = defaultdict(list)
    for log in logs:
        grouped[(log["shuffle_idx"], log["baseline"])].append(log)
        
    for (shuffle, baseline), run_logs in grouped.items():
        conflicts = [x for x in run_logs if x["expected_label"] != "clean"]
        cleans = [x for x in run_logs if x["expected_label"] == "clean"]
        
        # Retrieval Metrics
        precisions = []
        recalls = []
        exact_matches = 0
        
        label_detects = defaultdict(list)
        label_recalls = defaultdict(list)
        
        for c in conflicts:
            n_cand = len(c["candidate_docs"])
            n_found = c["n_implicated_found"]
            n_gold = len(c["implicates"])
            
            p = n_found / n_cand if n_cand > 0 else 1.0 if n_gold == 0 else 0.0
            r = n_found / n_gold if n_gold > 0 else 1.0 if n_cand == 0 else 0.0
            
            precisions.append(p)
            recalls.append(r)
            
            label = c["expected_label"]
            label_detects[label].append(1.0 if c["detected"] else 0.0)
            label_recalls[label].append(r)
            
            if set(c["candidate_docs"]) == set(c["implicates"]):
                exact_matches += 1
                
        mean_p = statistics.mean(precisions) if precisions else 0.0
        mean_r = statistics.mean(recalls) if recalls else 0.0
        f1 = 2 * mean_p * mean_r / (mean_p + mean_r) if (mean_p + mean_r) > 0 else 0.0
        
        fp_pressure = statistics.mean([x["n_candidates"] for x in cleans]) if cleans else 0.0
        
        metrics_per_run[baseline]["precision"].append(mean_p)
        metrics_per_run[baseline]["recall"].append(mean_r)
        metrics_per_run[baseline]["f1"].append(f1)
        metrics_per_run[baseline]["fp_pressure"].append(fp_pressure)
        metrics_per_run[baseline]["exact_match_rate"].append(exact_matches / len(conflicts))
        
        for lbl in label_detects:
            dr = statistics.mean(label_detects[lbl])
            ar = statistics.mean(label_recalls[lbl])
            by_label_per_run[baseline][lbl]["dr"].append(dr)
            by_label_per_run[baseline][lbl]["ar"].append(ar)

    # Aggregate across shuffles
    agg_metrics = {}
    for baseline in baselines:
        agg_metrics[baseline] = {
            "Precision": statistics.mean(metrics_per_run[baseline]["precision"]),
            "Recall": statistics.mean(metrics_per_run[baseline]["recall"]),
            "F1-Score": statistics.mean(metrics_per_run[baseline]["f1"]),
            "Exact Match Rate": statistics.mean(metrics_per_run[baseline]["exact_match_rate"]),
            "FP Pressure": statistics.mean(metrics_per_run[baseline]["fp_pressure"]),
            "Detect Rate": summary["aggregate"][baseline]["detect_rate"]["mean"],
            "Affected Recall": summary["aggregate"][baseline]["affected_recall"]["mean"],
            "by_label": {}
        }
        for lbl in by_label_per_run[baseline]:
            agg_metrics[baseline]["by_label"][lbl] = {
                "Detect Rate": statistics.mean(by_label_per_run[baseline][lbl]["dr"]),
                "Affected Recall": statistics.mean(by_label_per_run[baseline][lbl]["ar"])
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
        fp = f"{agg_metrics[b]['FP Pressure']:.3f}"
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

    md = "# Extended Metrics & LaTeX Tables\n\n"
    md += "This report presents the computed metrics and the pre-filled LaTeX tables requested in the paper.\n\n"
    
    md += "## Extended Retrieval Metrics\n"
    md += "| Baseline | Precision | Recall (Affected-Recall) | F1-Score | Exact Match Rate | Clean-Control FP Pressure |\n"
    md += "|---|---|---|---|---|---|\n"
    for b in baselines:
        m = agg_metrics[b]
        md += f"| **{b}** | {m['Precision']:.2%} | {m['Recall']:.2%} | {m['F1-Score']:.2%} | {m['Exact Match Rate']:.2%} | {m['FP Pressure']:.2f} |\n"
        
    md += "\n## LaTeX Tables to Copy into Paper\n"
    md += "Below are the pre-filled LaTeX tables for your paper.\n\n"
    md += "### Table 1: Corpus and event-stream summary\n"
    md += f"```latex{table1}```\n\n"
    
    md += "### Table 2: Main Comparison\n"
    md += f"```latex{table2}```\n\n"
    
    md += "### Table 3: Breakdown by Conflict Type\n"
    md += f"```latex{table3}```\n"
    
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(md)
        
    print("Done!")

if __name__ == "__main__":
    calculate_metrics()

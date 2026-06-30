"""
RAGermeister — Chart Generation Script
Generates all proposed charts from experiment logs.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from collections import defaultdict, Counter

# ── Config ──────────────────────────────────────────────────────────
LOG_DIR = Path(r"c:\STUDIA\Machine Learning - magister\NLP\Projekt Grupowy\ragermeister\logs")
FIG_DIR = Path(r"c:\STUDIA\Machine Learning - magister\NLP\Projekt Grupowy\ragermeister\figures")
FIG_DIR.mkdir(exist_ok=True)

SUMMARY_FILE = LOG_DIR / "run_20260630_092642_summary.json"
JSONL_FILE   = LOG_DIR / "run_20260630_092642.jsonl"

BASELINES = ["vector-only", "metadata-only", "graph-only", "hybrid"]
BASELINE_LABELS = ["Vector", "Metadata", "Graph", "Hybrid"]
BASELINE_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
CONFLICT_TYPES = ["contradiction", "duplicate", "supersedes"]
CONFLICT_LABELS = ["Contradiction", "Duplicate", "Supersedes"]

DPI = 300
sns.set_theme(style="whitegrid", font_scale=1.1, rc={"figure.dpi": DPI})

# ── Load data ───────────────────────────────────────────────────────
print("Loading summary...")
with open(SUMMARY_FILE, 'r', encoding='utf-8') as f:
    summary = json.load(f)

print("Loading JSONL logs...")
with open(JSONL_FILE, 'r', encoding='utf-8') as f:
    logs = [json.loads(line) for line in f]
print(f"  Loaded {len(logs)} log entries.")

per_shuffle = summary["per_shuffle"]
aggregate   = summary["aggregate"]
n_shuffles  = summary["meta"]["n_shuffles"]

# ════════════════════════════════════════════════════════════════════
# CHART 1: Main Comparison Bar Chart
# ════════════════════════════════════════════════════════════════════
def chart1_main_comparison():
    print("Generating Chart 1: Main Comparison...")
    metrics = ["detect_rate", "affected_recall"]
    metric_labels = ["Detect-Rate", "Affected-Recall"]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    
    x = np.arange(len(BASELINES))
    width = 0.6
    
    for ax, metric, mlabel in zip(axes, metrics, metric_labels):
        means = [aggregate[b][metric]["mean"] for b in BASELINES]
        stds  = [aggregate[b][metric]["std"]  for b in BASELINES]
        
        bars = ax.bar(x, means, width, yerr=stds, capsize=5,
                      color=BASELINE_COLORS, edgecolor='white', linewidth=0.5,
                      error_kw={'linewidth': 1.5})
        
        ax.set_ylabel(mlabel if ax == axes[0] else "")
        ax.set_title(mlabel, fontweight='bold', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(BASELINE_LABELS, fontsize=11)
        ax.set_ylim(0.65, 1.04)
        ax.set_yticks([0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
        
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2., mean + std + 0.01,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    fig.suptitle("Discovery Quality Across Baselines", fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_main_comparison.png", bbox_inches='tight', dpi=DPI)
    plt.close(fig)
    print("  -> 01_main_comparison.png")


# ════════════════════════════════════════════════════════════════════
# CHART 2: Detect-Rate by Conflict Type — Heatmap
# ════════════════════════════════════════════════════════════════════
def chart2_heatmap():
    print("Generating Chart 2: Heatmap...")
    data = np.zeros((len(BASELINES), len(CONFLICT_TYPES)))
    annot = np.empty_like(data, dtype=object)
    
    for i, b in enumerate(BASELINES):
        for j, ct in enumerate(CONFLICT_TYPES):
            mean = aggregate[b]["by_label"][ct]["detect_rate_mean"]
            std  = aggregate[b]["by_label"][ct]["detect_rate_std"]
            data[i, j] = mean
            annot[i, j] = f"{mean:.3f}\n+/-{std:.3f}"
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(data, annot=annot, fmt='', cmap='RdYlGn', vmin=0.6, vmax=1.0,
                xticklabels=CONFLICT_LABELS, yticklabels=BASELINE_LABELS,
                linewidths=2, linecolor='white', ax=ax,
                cbar_kws={'label': 'Detect-Rate (mean +/- std)'})
    ax.set_title("Detect-Rate by Conflict Type", fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_heatmap_by_type.png", bbox_inches='tight', dpi=DPI)
    plt.close(fig)
    print("  -> 02_heatmap_by_type.png")


# ════════════════════════════════════════════════════════════════════
# CHART 3: Detect-Rate by Conflict Type — Grouped Bars
# ════════════════════════════════════════════════════════════════════
def chart3_grouped_bars_by_type():
    print("Generating Chart 3: Grouped Bars by Type...")
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(CONFLICT_TYPES))
    width = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]
    
    for i, (b, label, color) in enumerate(zip(BASELINES, BASELINE_LABELS, BASELINE_COLORS)):
        means = [aggregate[b]["by_label"][ct]["detect_rate_mean"] for ct in CONFLICT_TYPES]
        stds  = [aggregate[b]["by_label"][ct]["detect_rate_std"]  for ct in CONFLICT_TYPES]
        ax.bar(x + offsets[i]*width, means, width, yerr=stds, capsize=4,
               label=label, color=color, edgecolor='white', linewidth=0.5,
               error_kw={'linewidth': 1.2})
    
    ax.set_ylabel("Detect-Rate", fontsize=12)
    ax.set_title("Detect-Rate by Conflict Type (mean +/- std over 5 shuffles)",
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(CONFLICT_LABELS, fontsize=12)
    ax.set_ylim(0.55, 1.08)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
    ax.legend(loc='lower left', framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_grouped_bars_by_type.png", bbox_inches='tight', dpi=DPI)
    plt.close(fig)
    print("  -> 03_grouped_bars_by_type.png")


# ════════════════════════════════════════════════════════════════════
# CHART 4: Temporal Detection — Line Chart (Q1-Q4)
# ════════════════════════════════════════════════════════════════════
def chart4_temporal():
    print("Generating Chart 4: Temporal Detection...")
    quarters = ["Q1\n(pos 0-62)", "Q2\n(pos 63-124)", "Q3\n(pos 125-186)", "Q4\n(pos 187-250)"]
    q_keys = ["Q1_pos_0-62", "Q2_pos_62-124", "Q3_pos_124-186", "Q4_pos_186-250"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for b, label, color in zip(BASELINES, BASELINE_LABELS, BASELINE_COLORS):
        rates_per_shuffle = []
        for sh in per_shuffle:
            rates = [sh["metrics"][b]["temporal_detection"][qk]["rate"] for qk in q_keys]
            rates_per_shuffle.append(rates)
        
        rates_arr = np.array(rates_per_shuffle)
        means = rates_arr.mean(axis=0)
        mins  = rates_arr.min(axis=0)
        maxs  = rates_arr.max(axis=0)
        
        x = np.arange(len(quarters))
        ax.plot(x, means, 'o-', label=label, color=color, linewidth=2.5, markersize=8)
        ax.fill_between(x, mins, maxs, color=color, alpha=0.12)
    
    ax.set_xticks(np.arange(len(quarters)))
    ax.set_xticklabels(quarters, fontsize=11)
    ax.set_ylabel("Detect-Rate", fontsize=12)
    ax.set_title("Detection Rate Along the Event Stream\n(lines = mean, bands = min-max over 5 shuffles)",
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0.25, 1.08)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
    ax.legend(loc='lower right', framealpha=0.9, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_temporal_detection.png", bbox_inches='tight', dpi=DPI)
    plt.close(fig)
    print("  -> 04_temporal_detection.png")


# ════════════════════════════════════════════════════════════════════
# CHART 5: Detect-Rate by Intent (Fine-Grained)
# ════════════════════════════════════════════════════════════════════
def chart5_by_intent():
    print("Generating Chart 5: By Intent...")
    intent_stats = defaultdict(lambda: defaultdict(list))
    
    for entry in logs:
        if entry["expected_label"] == "clean":
            continue
        intent = entry["intent"]
        baseline = entry["baseline"]
        intent_stats[intent][baseline].append(1 if entry["detected"] else 0)
    
    intent_order = [
        "direct_supersession",
        "entity_mediated_contradiction",
        "transitive_supersession_chain",
        "supersession_with_retroactive_staleness",
        "stale_artifact",
        "duplicate",
    ]
    intent_labels = [
        "Direct\nSupersession",
        "Entity-Mediated\nContradiction",
        "Transitive\nChain",
        "Retroactive\nStaleness",
        "Stale\nArtifact",
        "Duplicate",
    ]
    
    valid_intents = [i for i in intent_order if i in intent_stats]
    valid_labels  = [intent_labels[intent_order.index(i)] for i in valid_intents]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(valid_intents))
    width = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]
    
    for i, (b, label, color) in enumerate(zip(BASELINES, BASELINE_LABELS, BASELINE_COLORS)):
        rates = []
        for intent in valid_intents:
            vals = intent_stats[intent].get(b, [])
            rates.append(np.mean(vals) if vals else 0)
        ax.bar(x + offsets[i]*width, rates, width, label=label, color=color,
               edgecolor='white', linewidth=0.5)
    
    ax.set_ylabel("Detect-Rate", fontsize=12)
    ax.set_title("Detect-Rate by Event Intent (Fine-Grained Breakdown)",
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_labels, fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
    ax.legend(loc='lower center', ncol=4, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_by_intent.png", bbox_inches='tight', dpi=DPI)
    plt.close(fig)
    print("  -> 05_by_intent.png")


# ════════════════════════════════════════════════════════════════════
# CHART 6: FP Pressure Distribution — Violin + Box
# ════════════════════════════════════════════════════════════════════
def chart6_fp_pressure():
    print("Generating Chart 6: FP Pressure...")
    pressure_data = defaultdict(list)
    
    for entry in logs:
        if entry["expected_label"] != "clean":
            continue
        pressure_data[entry["baseline"]].append(entry["n_candidates"])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    plot_data = [pressure_data[b] for b in BASELINES]
    parts = ax.violinplot(plot_data, positions=range(len(BASELINES)),
                          showmeans=True, showmedians=True, showextrema=False)
    
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(BASELINE_COLORS[i])
        pc.set_alpha(0.6)
    parts['cmeans'].set_color('black')
    parts['cmedians'].set_color('darkred')
    
    # Overlay individual points (jittered)
    for i, b in enumerate(BASELINES):
        vals = pressure_data[b]
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=3, color=BASELINE_COLORS[i], alpha=0.15)
    
    ax.set_xticks(range(len(BASELINES)))
    
    new_labels = []
    for b, label in zip(BASELINES, BASELINE_LABELS):
        m = np.mean(pressure_data[b])
        new_labels.append(f"{label}\n(mean={m:.2f})")
    ax.set_xticklabels(new_labels, fontsize=11)
    
    ax.set_ylabel("# Candidates on Clean Events", fontsize=12)
    ax.set_title("False-Positive Candidate Pressure on Clean Events\n(lower = less work for the judge)",
                 fontsize=13, fontweight='bold')
    
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_fp_pressure_violin.png", bbox_inches='tight', dpi=DPI)
    plt.close(fig)
    print("  -> 06_fp_pressure_violin.png")


# ════════════════════════════════════════════════════════════════════
# CHART 7: Cumulative Detect-Rate (Running Average)
# ════════════════════════════════════════════════════════════════════
def chart7_running_average():
    print("Generating Chart 7: Running Average...")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for b, label, color in zip(BASELINES, BASELINE_LABELS, BASELINE_COLORS):
        all_running = []
        for sh_idx in range(n_shuffles):
            entries = sorted(
                [e for e in logs if e["baseline"] == b and e["shuffle_idx"] == sh_idx
                 and e["expected_label"] != "clean"],
                key=lambda e: e["position"]
            )
            if not entries:
                continue
            cumsum = np.cumsum([1 if e["detected"] else 0 for e in entries])
            running = cumsum / np.arange(1, len(cumsum) + 1)
            all_running.append(running)
        
        min_len = min(len(r) for r in all_running)
        arr = np.array([r[:min_len] for r in all_running])
        means = arr.mean(axis=0)
        mins  = arr.min(axis=0)
        maxs  = arr.max(axis=0)
        
        x = np.arange(1, min_len + 1)
        ax.plot(x, means, '-', label=label, color=color, linewidth=2)
        ax.fill_between(x, mins, maxs, color=color, alpha=0.1)
    
    ax.set_xlabel("Conflict Event # (chronological order in stream)", fontsize=12)
    ax.set_ylabel("Cumulative Detect-Rate", fontsize=12)
    ax.set_title("Running Average Detect-Rate Over Conflict Events\n(lines = mean, bands = min-max over 5 shuffles)",
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0.55, 1.05)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
    ax.legend(loc='lower right', framealpha=0.9, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_running_average.png", bbox_inches='tight', dpi=DPI)
    plt.close(fig)
    print("  -> 07_running_average.png")


# ════════════════════════════════════════════════════════════════════
# CHART 8: Graph Growth — Area Chart
# ════════════════════════════════════════════════════════════════════
def chart8_graph_growth():
    print("Generating Chart 8: Graph Growth...")
    entries = sorted(
        [e for e in logs if e["baseline"] == "graph-only" and e["shuffle_idx"] == 0 and e.get("graph")],
        key=lambda e: e["position"]
    )
    
    positions = [e["position"] for e in entries]
    chunks   = [e["graph"]["nodes"]["chunk"]["active"] for e in entries]
    entities = [e["graph"]["nodes"]["entity"]["active"] for e in entries]
    topics   = [e["graph"]["nodes"]["topic"]["active"] for e in entries]
    dates    = [e["graph"]["nodes"]["date"]["active"] for e in entries]
    structural = [e["graph"]["edges"]["structural"] for e in entries]
    semantic_a = [e["graph"]["edges"]["semantic_active"] for e in entries]
    semantic_r = [e["graph"]["edges"]["semantic_revoked"] for e in entries]
    tombstoned = [e["graph"]["nodes"]["chunk"]["tombstoned"] for e in entries]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 10), sharex=True)
    
    ax1.stackplot(positions,
                  chunks, entities, topics, dates,
                  labels=["Chunks (active)", "Entities", "Topics", "Dates"],
                  colors=["#4C72B0", "#C44E52", "#55A868", "#CCB974"],
                  alpha=0.8)
    ax1.plot(positions, tombstoned, '--', color='black', linewidth=1.5, label="Chunks (tombstoned)")
    ax1.set_ylabel("Node Count", fontsize=12)
    ax1.set_title("Knowledge Graph Growth - Nodes", fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
    
    ax2.stackplot(positions,
                  structural, semantic_a, semantic_r,
                  labels=["Structural", "Semantic (active)", "Semantic (revoked)"],
                  colors=["#4C72B0", "#55A868", "#C44E52"],
                  alpha=0.8)
    ax2.set_xlabel("Stream Position", fontsize=12)
    ax2.set_ylabel("Edge Count", fontsize=12)
    ax2.set_title("Knowledge Graph Growth - Edges", fontsize=13, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10, framealpha=0.9)
    
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_graph_growth.png", bbox_inches='tight', dpi=DPI)
    plt.close(fig)
    print("  -> 08_graph_growth.png")


# ════════════════════════════════════════════════════════════════════
# CHART 9: Recall vs Pressure Trade-off — Scatter
# ════════════════════════════════════════════════════════════════════
def chart9_tradeoff():
    print("Generating Chart 9: Recall vs Pressure...")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for b, label, color in zip(BASELINES, BASELINE_LABELS, BASELINE_COLORS):
        dr_mean = aggregate[b]["detect_rate"]["mean"]
        dr_std  = aggregate[b]["detect_rate"]["std"]
        fp_mean = aggregate[b]["fp_pressure_mean"]["mean"]
        fp_std  = aggregate[b]["fp_pressure_mean"]["std"]
        
        # Add slight jitter for overlapping points (Vector and Hybrid)
        if b == "vector-only":
            fp_mean -= 0.02
            dr_mean -= 0.005
        elif b == "hybrid":
            fp_mean += 0.02
            dr_mean += 0.005
            
        ax.errorbar(fp_mean, dr_mean, xerr=fp_std, yerr=dr_std,
                    fmt='o', color=color, markersize=14, capsize=8,
                    markeredgecolor='white', markeredgewidth=2, linewidth=2.5,
                    label=label, zorder=5)
                    
        # Annotate each point to make them easier to distinguish
        xytext_offset = (20, -20) if b == "vector-only" else (-20, 20) if b == "hybrid" else (20, 20)
        ax.annotate(label, xy=(fp_mean, dr_mean), xytext=xytext_offset, 
                    textcoords='offset points', fontsize=10, fontweight='bold', color=color,
                    arrowprops=dict(arrowstyle="->", color=color, alpha=0.6))
    
    ax.set_xlabel("FP Candidate Pressure (mean candidates on clean events)", fontsize=12)
    ax.set_ylabel("Detect-Rate", fontsize=12)
    ax.set_title("Recall vs. False-Positive Pressure Trade-off",
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0.7, 1.05)
    ax.set_xlim(3.4, 5.4)
    # The legend can be removed since we have direct annotations now, or we can keep it. We'll remove it to reduce clutter.
    
    ax.text(3.45, 1.02, "Note: Crosshairs indicate standard deviation (\u00b1std) over 5 shuffles.\nVector and Hybrid have near-zero variance in FP pressure,\nmaking their horizontal error bars invisible.", 
            fontsize=9, style='italic', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'), va='top')
            
    fig.tight_layout()
    fig.savefig(FIG_DIR / "09_recall_vs_pressure.png", bbox_inches='tight', dpi=DPI)
    plt.close(fig)
    print("  -> 09_recall_vs_pressure.png")


# ════════════════════════════════════════════════════════════════════
# CHART 10: Radar / Spider Chart
# ════════════════════════════════════════════════════════════════════
def chart10_radar():
    print("Generating Chart 10: Radar Chart...")
    categories = ["Detect-Rate", "Affected-Recall", "Contradiction\nRate",
                   "Supersedes\nRate", "Duplicate\nRate", "Low FP\nPressure"]
    N = len(categories)
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    
    max_fp = max(aggregate[b]["fp_pressure_mean"]["mean"] for b in BASELINES)
    
    for b, label, color in zip(BASELINES, BASELINE_LABELS, BASELINE_COLORS):
        values = [
            aggregate[b]["detect_rate"]["mean"],
            aggregate[b]["affected_recall"]["mean"],
            aggregate[b]["by_label"]["contradiction"]["detect_rate_mean"],
            aggregate[b]["by_label"]["supersedes"]["detect_rate_mean"],
            aggregate[b]["by_label"]["duplicate"]["detect_rate_mean"],
            1 - (aggregate[b]["fp_pressure_mean"]["mean"] / (max_fp + 1)),
        ]
        values += values[:1]
        linestyle = '--' if b == "vector-only" else '-'
        ax.plot(angles, values, color=color, linewidth=2, linestyle=linestyle, marker='o', markersize=5, label=label)
        ax.fill(angles, values, alpha=0.08, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_title("Multi-Metric Baseline Comparison", fontsize=14, fontweight='bold', pad=20)
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=color, lw=2, linestyle='--' if b == "vector-only" else '-', label=label)
        for b, label, color in zip(BASELINES, BASELINE_LABELS, BASELINE_COLORS)
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10, framealpha=0.9)
    
    fig.tight_layout()
    fig.savefig(FIG_DIR / "10_radar_chart.png", bbox_inches='tight', dpi=DPI)
    plt.close(fig)
    print("  -> 10_radar_chart.png")


# ════════════════════════════════════════════════════════════════════
# CHART 11: Misses per Baseline — Heatmap
# ════════════════════════════════════════════════════════════════════
def chart11_misses():
    print("Generating Chart 11: Misses Heatmap...")
    conflict_doc_ids = set()
    detection_counts = defaultdict(lambda: defaultdict(int))
    
    for entry in logs:
        if entry["expected_label"] == "clean":
            continue
        doc_id = entry["doc_id"]
        conflict_doc_ids.add(doc_id)
        if entry["detected"]:
            detection_counts[entry["baseline"]][doc_id] += 1
    
    interesting_docs = []
    for doc_id in sorted(conflict_doc_ids):
        for b in BASELINES:
            if detection_counts[b].get(doc_id, 0) < n_shuffles:
                interesting_docs.append(doc_id)
                break
    
    if not interesting_docs:
        print("  -> No misses found, skipping chart 11.")
        return
    
    miss_score = {}
    for doc_id in interesting_docs:
        miss_score[doc_id] = sum(n_shuffles - detection_counts[b].get(doc_id, 0) for b in BASELINES)
    interesting_docs = sorted(interesting_docs, key=lambda d: -miss_score[d])[:30]
    
    data = np.zeros((len(BASELINES), len(interesting_docs)))
    for i, b in enumerate(BASELINES):
        for j, doc_id in enumerate(interesting_docs):
            data[i, j] = detection_counts[b].get(doc_id, 0)
    
    fig, ax = plt.subplots(figsize=(max(14, len(interesting_docs) * 0.6), 5))
    sns.heatmap(data, annot=True, fmt='.0f', cmap='RdYlGn', vmin=0, vmax=n_shuffles,
                xticklabels=interesting_docs, yticklabels=BASELINE_LABELS,
                linewidths=1, linecolor='white', ax=ax,
                cbar_kws={'label': f'# Shuffles Detected (out of {n_shuffles})'})
    ax.set_title("Detection Reliability per Event - Which Events Are Hard?",
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "11_misses_heatmap.png", bbox_inches='tight', dpi=DPI)
    plt.close(fig)
    print("  -> 11_misses_heatmap.png")


# ════════════════════════════════════════════════════════════════════
# CHART 12: Entity Cluster Growth
# ════════════════════════════════════════════════════════════════════
def chart12_entity_clusters():
    print("Generating Chart 12: Entity Clusters...")
    last_entry = sorted(
        [e for e in logs if e["baseline"] == "graph-only" and e["shuffle_idx"] == 0 and e.get("graph")],
        key=lambda e: e["position"]
    )[-1]
    
    clusters = last_entry["graph"]["entity_clusters"]
    clusters = sorted(clusters, key=lambda c: -c["active_chunk_count"])[:15]
    
    names  = [c["name"][:25] for c in clusters]
    counts = [c["active_chunk_count"] for c in clusters]
    types  = [c["type"] for c in clusters]
    
    type_colors = {"person": "#C44E52", "org": "#4C72B0"}
    colors = [type_colors.get(t, "#888888") for t in types]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(range(len(names)), counts, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Active Chunk Count", fontsize=12)
    ax.set_title("Top-15 Entity Clusters by Active Chunk Count (end of stream)",
                 fontsize=13, fontweight='bold')
    
    legend_elements = [mpatches.Patch(facecolor=v, label=k.capitalize())
                       for k, v in type_colors.items()]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11, framealpha=0.9)
    
    fig.tight_layout()
    fig.savefig(FIG_DIR / "12_entity_clusters.png", bbox_inches='tight', dpi=DPI)
    plt.close(fig)
    print("  -> 12_entity_clusters.png")


# ════════════════════════════════════════════════════════════════════
# CHART 13: Candidate Overlap Between Baselines
# ════════════════════════════════════════════════════════════════════
def chart13_candidate_overlap():
    print("Generating Chart 13: Candidate Overlap...")
    conflict_entries = defaultdict(dict)
    for entry in logs:
        if entry["shuffle_idx"] != 0 or entry["expected_label"] == "clean":
            continue
        key = (entry["position"], entry["doc_id"])
        conflict_entries[key][entry["baseline"]] = set(entry["candidate_docs"])
    
    pairs = []
    for i, b1 in enumerate(BASELINES):
        for j, b2 in enumerate(BASELINES):
            if i >= j:
                continue
            jaccards = []
            for key, baselines_cands in conflict_entries.items():
                if b1 in baselines_cands and b2 in baselines_cands:
                    s1 = baselines_cands[b1]
                    s2 = baselines_cands[b2]
                    if s1 or s2:
                        jaccards.append(len(s1 & s2) / len(s1 | s2))
            if jaccards:
                pairs.append((b1, b2, np.mean(jaccards), np.std(jaccards)))
    
    exclusive = defaultdict(list)
    for key, baselines_cands in conflict_entries.items():
        all_sets = {b: baselines_cands.get(b, set()) for b in BASELINES}
        for b in BASELINES:
            others = set().union(*(all_sets[ob] for ob in BASELINES if ob != b))
            exc = all_sets[b] - others
            exclusive[b].append(len(exc))
    
    n = len(BASELINES)
    jaccard_matrix = np.eye(n)
    for b1, b2, jmean, jstd in pairs:
        i = BASELINES.index(b1)
        j = BASELINES.index(b2)
        jaccard_matrix[i, j] = jmean
        jaccard_matrix[j, i] = jmean
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.heatmap(jaccard_matrix, annot=True, fmt='.3f', cmap='YlOrRd', vmin=0, vmax=1,
                xticklabels=BASELINE_LABELS, yticklabels=BASELINE_LABELS,
                linewidths=2, linecolor='white', ax=ax1)
    ax1.set_title("Candidate Set Overlap\n(mean Jaccard on conflict events)", fontsize=12, fontweight='bold')
    
    exc_means = [np.mean(exclusive[b]) for b in BASELINES]
    ax2.bar(range(n), exc_means, color=BASELINE_COLORS, edgecolor='white', linewidth=0.5)
    ax2.set_xticks(range(n))
    ax2.set_xticklabels(BASELINE_LABELS, fontsize=11)
    ax2.set_ylabel("Mean Exclusive Candidates\n(not found by any other baseline)", fontsize=11)
    ax2.set_title("Unique Contribution per Baseline\n(conflict events, shuffle 0)", fontsize=12, fontweight='bold')
    
    fig.suptitle("Candidate Discovery Complementarity", fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "13_candidate_overlap.png", bbox_inches='tight', dpi=DPI)
    plt.close(fig)
    print("  -> 13_candidate_overlap.png")


# ════════════════════════════════════════════════════════════════════
# RUN ALL
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("RAGermeister - Chart Generation")
    print("=" * 60)
    
    chart1_main_comparison()
    chart2_heatmap()
    chart3_grouped_bars_by_type()
    chart4_temporal()
    chart5_by_intent()
    chart6_fp_pressure()
    chart7_running_average()
    chart8_graph_growth()
    chart9_tradeoff()
    chart10_radar()
    chart11_misses()
    chart12_entity_clusters()
    chart13_candidate_overlap()
    
    print("\n" + "=" * 60)
    print(f"All charts saved to: {FIG_DIR}")
    print("=" * 60)

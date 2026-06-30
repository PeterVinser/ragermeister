# Extended Metrics & LaTeX Tables

Source run: `run_20260630_092642.jsonl` (5 shuffles).

This report presents the computed metrics and the pre-filled LaTeX tables requested in the paper.

## Discovery Metrics
_**Candidate yield** = fraction of surfaced candidates that are gold-implicated — an on-target-vs-noisy index, **not precision**: a surfaced non-implicated doc is not a false positive (the judge filters it, and under the oracle the real FP rate is 0). **Candidate pressure** = mean #candidates on clean events (a count, **not** a false-alert rate). The headline quality metrics are detect-rate, affected-recall and macro-F1._

| Baseline | Detect-Rate | Affected-Recall | Macro-F1 | Candidate Yield | Candidate Pressure (clean) |
|---|---|---|---|---|---|
| **vector-only** | 98.96% | 98.88% | 0.996 | 22.86% | 4.91 |
| **metadata-only** | 90.13% | 86.07% | 0.940 | 24.15% | 3.94 |
| **graph-only** | 78.70% | 77.98% | 0.901 | 18.03% | 4.33 |
| **hybrid** | 99.74% | 99.55% | 0.998 | 23.01% | 4.91 |

## Status-Classification Macro-F1
Predicted status (`verdict`) vs gold (`expected_label`) over all events. **Caveat:** these logs use the OracleJudge, which never confuses one conflict class for another, so conflict-class precision is 1.0 by construction and macro-F1 tracks recall. Re-run with the real `ConflictJudge` for an end-to-end number.

| Baseline | Macro-F1 (4-class) | Macro-F1 (conflict-only) | F1 clean | F1 duplicate | F1 contradiction | F1 supersedes |
|---|---|---|---|---|---|---|
| **vector-only** | 0.996 ± 0.005 | 0.995 | 0.998 | 1.000 | 0.992 | 0.993 |
| **metadata-only** | 0.940 ± 0.014 | 0.927 | 0.979 | 0.980 | 0.838 | 0.961 |
| **graph-only** | 0.901 ± 0.023 | 0.883 | 0.955 | 0.922 | 0.865 | 0.861 |
| **hybrid** | 0.998 ± 0.004 | 0.997 | 0.999 | 1.000 | 0.992 | 1.000 |

## LaTeX Tables to Copy into Paper
Below are the pre-filled LaTeX tables for your paper.

### Table 1: Corpus and event-stream summary
```latex
\begin{table}[h]
\centering
\begin{tabular}{ll}
\toprule
Property & Value \\
\midrule
Initial documents & -- \\
Total change events & 250 \\
Conflict events (contradiction / duplicate / supersede) & 77 (13 / 21 / 43) \\
Clean events & 173 \\
Domain & -- \\
Embedding model & text-embedding-3-large \\
\bottomrule
\end{tabular}
\caption{Corpus and event-stream summary.}
\end{table}
```

### Table 2: Main Comparison
```latex
\begin{table}[h]
\centering
\begin{tabular}{lccc}
\toprule
Baseline & detect-rate & affected-recall & clean-control pressure \\
\midrule
Vector-only & 0.990 & 0.989 & 4.913 \\
Metadata-only & 0.901 & 0.861 & 3.935 \\
Graph-only & 0.787 & 0.780 & 4.332 \\
Hybrid & 0.997 & 0.996 & 4.913 \\
\bottomrule
\end{tabular}
\caption{Discovery quality across baselines (default configuration).}
\end{table}
```

### Table 3: Breakdown by Conflict Type
```latex
\begin{table}[h]
\centering
\begin{tabular}{lcccc}
\toprule
Conflict type & Vector & Metadata & Graph & Hybrid \\
\midrule
Contradiction (DR) & 0.985 & 0.723 & 0.769 & 0.985 \\
Contradiction (AR) & 0.977 & 0.646 & 0.692 & 0.977 \\
Duplicate (DR) & 1.000 & 0.962 & 0.857 & 1.000 \\
Duplicate (AR) & 1.000 & 0.962 & 0.857 & 1.000 \\
Supersede (DR) & 0.986 & 0.926 & 0.758 & 1.000 \\
Supersede (AR) & 0.986 & 0.902 & 0.747 & 1.000 \\
\bottomrule
\end{tabular}
\caption{detect-rate (DR) and affected-recall (AR) by conflict type.}
\end{table}
```

### Table 4: Status-Classification Macro-F1
```latex
\begin{table}[h]
\centering
\begin{tabular}{lccccc}
\toprule
Baseline & Macro-F1 & clean & duplicate & contradiction & supersedes \\
\midrule
Vector-only & 0.996 & 0.998 & 1.000 & 0.992 & 0.993 \\
Metadata-only & 0.940 & 0.979 & 0.980 & 0.838 & 0.961 \\
Graph-only & 0.901 & 0.955 & 0.922 & 0.865 & 0.861 \\
Hybrid & 0.998 & 0.999 & 1.000 & 0.992 & 1.000 \\
\bottomrule
\end{tabular}
\caption{Status-classification macro-F1 and per-class F1 (oracle judge; conflict-class precision is 1.0 by construction).}
\end{table}
```

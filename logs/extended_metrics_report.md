# Extended Metrics & LaTeX Tables

This report presents the computed metrics and the pre-filled LaTeX tables requested in the paper.

## Extended Retrieval Metrics
| Baseline | Precision | Recall (Affected-Recall) | F1-Score | Exact Match Rate | Clean-Control FP Pressure |
|---|---|---|---|---|---|
| **vector-only** | 22.86% | 98.83% | 37.13% | 0.00% | 4.91 |
| **metadata-only** | 24.15% | 87.53% | 37.85% | 2.86% | 3.94 |
| **graph-only** | 18.03% | 76.75% | 29.19% | 0.00% | 4.33 |
| **hybrid** | 23.01% | 99.61% | 37.39% | 0.00% | 4.91 |

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

# CHANGES — metryki ewaluacji (macro-F1, naprawa metryk, koszt/runtime)

Zmiany w warstwie **ewaluacji i metryk**. Nie ruszają logiki baseline'ów ani danych —
dotykają tylko tego, *co i jak mierzymy i raportujemy*. Run odniesienia: `run_20260630_092642`
(v3, 250 zdarzeń, 5 shuffli, `--llm-extractor`, OracleJudge, insert-only).

Branch: `mslowakiewicz`.

---

## 1. Macro-F1 po statusach (NOWE)

Dodane liczenie **status-classification macro-F1** + confusion matrix + per-class P/R/F1.
Predykcja = `verdict` z rekordu logu, gold = `expected_label`. Działa na istniejących
logach (bez re-runu) i uogólnia się na prawdziwego sędziego.

- **Gdzie:** `scripts/calculate_extended_metrics.py` → funkcja `macro_f1_for_run()`.
- **Wynik (mean ± std po shufflach):**

  | Baseline | macro-F1 (4 klasy) | conflict-only |
  |---|---|---|
  | Hybrid | 0.998 ± 0.004 | 0.997 |
  | Vector | 0.996 ± 0.005 | 0.995 |
  | Metadata | 0.940 ± 0.014 | 0.927 |
  | Graph | 0.901 ± 0.023 | 0.883 |

- **W logach:** doszły pola `Macro-F1`, `Macro-F1 (conflict-only)`, `status_per_class_f1`,
  pełna `confusion_matrix_total` (gold→pred, gotowa pod heatmapę) w `logs/extended_metrics.json`;
  tabela + **LaTeX Table 4** w `logs/extended_metrics_report.md`.

> **Caveat (wpisany też w raport):** te logi używają OracleJudge, który nie myli klas
> konfliktowych między sobą → precyzja klas konfliktowych = 1.0 z definicji, więc macro-F1
> jest **górną granicą napędzaną recall-em**, nie end-to-end. Confusion matrix potwierdza:
> jedyne błędy to konflikt→clean (przegapienia). Prawdziwy macro-F1 wymaga przebiegu z
> `ConflictJudge` (gpt-5.4) — kod już to obsłuży.

## 2. Naprawa mylących metryk (Precision / F1 / Exact-Match)

Stare „Precision/F1-Score/Exact-Match" w `calculate_extended_metrics.py` liczyły
`znalezione_implikowane / wszyscy_kandydaci` (≈0.23 / 0.37 / 0.00) — to **nie precyzja**
(housekeeper *ma* podać kilku kandydatów, sędzia filtruje resztę; pod oracle realny FP rate = 0).
Wpisanie ich do raportu obok „macro-F1 0.99" wyglądałoby jak błąd.

- **„Precision" → „Candidate Yield"** (odsetek wyciągniętych kandydatów trafionych w gold) z jawną definicją, że to **nie** precyzja.
- **„FP Pressure" → „Candidate Pressure (clean)"** z adnotacją, że to *liczność* kandydatów na clean-eventach, **nie** false-alert rate.
- **Usunięte:** retrieval „F1-Score" (mieszał fałszywą precyzję z recall) i „Exact Match Rate" (≈0 przez top-k).
- **Tabela w raporcie przebudowana** — prowadzą poprawne metryki: **detect-rate / affected-recall / macro-F1**; yield/pressure zdemonstrowane jako pomocnicze z caveatem.

## 3. Logowanie kosztu / runtime (NOWE)

Per-baseline koszt strony *discovery*: wywołania LLM, tokeny (in/out), wywołania embeddera, czas.

- **Nowy moduł `solution/services/cost_meter.py`** — thread-local rejestr. Instrumentowane są
  tylko dwie klasy-liście; **wszyscy wołający liczą się automatycznie**, a atrybucja idzie po
  wątku — więc nawet **współdzielony** `extractor`/`embedder` przy **równoległych** baseline'ach
  trafia do właściwego (zweryfikowane testem przy wymuszonym przeplocie wątków).
- **Instrumentacja:**
  - `solution/services/embedder.py` → `embed_calls`, `embed_texts`
  - `solution/services/llm.py` → `llm_calls`, `llm_tokens_in`, `llm_tokens_out` (oba wywołania)
- **`solution/eval/comparison.py`:**
  - `_run_all_baselines_parallel` — reset metera, każdy thunk taguje swój wątek + mierzy własny
    czas, po złączeniu zbiera koszt per baseline (sygnatura zwraca teraz też `cost_by_baseline`).
  - ścieżka `--shuffles 0` (sekwencyjna) też instrumentowana.
  - `aggregate_summaries` składa koszt (mean ± std po shufflach) → `summary.json`, klucz `cost`.
- **`calculate_extended_metrics.py`** — czyta koszt i renderuje **tabelę „Cost / Runtime"** +
  **LaTeX Table 5** (warunkowo: stary log bez kosztu jest grzecznie pomijany).

> **Uwaga:** koszt **nie da się odzyskać** ze starego logu — wymaga ponownego przebiegu z Azure.
> To koszt *discovery* (OracleJudge nie woła LLM); to właściwa oś porównania, bo różnica między
> baseline'ami siedzi w ekstrakcji encji + ER. Oczekiwany obraz: vector ~0 wywołań LLM, graf/hybryda
> ~300+ wywołań i kilkukrotnie dłużej → domyka narrację „graf gorszy **i** droższy".

## 4. Przenośność skryptu metryk

`calculate_extended_metrics.py` miał zahardkodowane ścieżki Windows (`c:\STUDIA\...`).
Teraz **auto-wykrywa najnowszy** `logs/run_*_summary.json` (+ pasujący `.jsonl`), z opcją
podania ścieżki z CLI. Działa na każdej maszynie.

---

## Jak odpalić

```bash
# pełny przebieg (wymaga klucza Azure dla --llm-extractor)
python -m solution.eval.comparison --input data/extended_events_v3.jsonl --shuffles 5 --llm-extractor

# metryki + raport (działa też na istniejącym logu; macro-F1 policzy bez Azure,
# koszt pojawi się tylko jeśli run był instrumentowany)
python scripts/calculate_extended_metrics.py            # auto: najnowszy log
python scripts/calculate_extended_metrics.py logs/run_<id>_summary.json   # konkretny
```

## Pliki

| Plik | Status | Co |
|---|---|---|
| `solution/services/cost_meter.py` | nowy | thread-local rejestr kosztu |
| `solution/services/embedder.py` | zmiana | liczniki embeddingów |
| `solution/services/llm.py` | zmiana | liczniki wywołań/tokenów LLM |
| `solution/eval/comparison.py` | zmiana | zbieranie + agregacja kosztu w summary |
| `scripts/calculate_extended_metrics.py` | zmiana | macro-F1, naprawa metryk, tabela kosztu, przenośne ścieżki |
| `logs/extended_metrics.json`, `logs/extended_metrics_report.md` | regenerowane | z macro-F1 (koszt po re-runie) |

## Co jeszcze warto (nie zrobione)

- **Przebieg z prawdziwym `ConflictJudge`** — da end-to-end macro-F1 (z pomyłkami między klasami),
  realny false-alert rate i koszt sędziego. Jedyny brakujący eksperyment do pełnego raportu.
- **Wykresy macro-F1 / confusion-matrix** — `confusion_matrix_total` jest już w JSON, ale
  `generate_charts.py` jeszcze tego nie rysuje.
- **Jakość entity-resolution** (ER precision/recall vs gold) — wyjaśnia, *dlaczego* graf przegrywa.

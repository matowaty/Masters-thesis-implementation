# Źródła wykresów, tabel i liczb w pracy (provenance)

Stan: 2026-09-20. Plik opisuje, **jaki skrypt wygenerował każdy rysunek i każdą tabelę** w `Masters-thesis-document/thesis/`, z jakich danych, oraz jak to odtworzyć. Wszystko, co jest w pracy poza wymienionymi niżej wyjątkami, powstaje z kodu zapisanego w tym folderze; żadnej liczby nie wpisano ręcznie.

## 1. Gdzie jest kod i dane

| Co | Gdzie (folder `Thesis - 2`) | Uwagi |
|---|---|---|
| Skrypty analiz (wykresy, tabele, liczby) | `thesis_analysis/` (`an_*.py`, `common.py`, `run_all.sh`, `inputs/`) | 12 skryptów; wspólne funkcje w `common.py` |
| Logi 8 pełnych przebiegów GA (A100) | `RESULTS/` (od 2026-09-20 jedyna kopia; 17 duplikatów z `A100-results/RESULTS`, porównanych rekurencyjnie plik po pliku z `RESULTS/`, przeniesiono do `_do_usuniecia_duplikaty_A100/`) | wejście dla `an_runs`, `an_v2_skill`, `an_magnitude`, `an_economics`, `an_ga`, `an_early`; tylko do odczytu |
| Manifest wszystkich 47 folderów wyników | `thesis_analysis/inputs/run_manifest_all.csv` (folder, data, maszyna, grupa, status w pracy, miejsce w pracy, checkpoint) | utworzony 2026-09-20 skryptem pomocniczym; podstawa planowanej tabeli inwentarza (`PLAN_INTEGRACJI_WYNIKOW.md`) |
| Surowe dane (LSEG, 5 min, 19 spółek) | `DATA/` | licencja LSEG/Refinitiv: nie publikować |
| Szybkie przebiegi V1 z ziarnami (RQ2, RQ4) | `RESULTS/` (foldery z `seed.json`), skrypt `quick_runs.py` | CPU, torch 2.14 |
| Przebieg V2 z przesunięciem okna (sekcja 7.8) | `RESULTS_V2SHIFT/` (12 plików), skrypt `quick_v2_shift.py` | CPU, ziarna 1-3, warianty `base` i `shift` |
| Test O1 (Model 2: klasyfikator i rozmiar zbioru kalibracyjnego; sekcja 7.5) | `RESULTS_O1/` (4 pliki `.jsonl`: `{fast,full}_part{0,1}.jsonl`), skrypty `quick_o1_model2.py` i `run_o1.sh` | CPU, 2 rdzenie; losowa populacja 20 osobników (ziarno 20260920), 2 tryby × 4 warianty Model 2; wejście dla `an_o1`, zmienna `O1_RESULTS` |
| Wczesne przebiegi V2 (16-18.06) | `RESULTS/` (foldery E-1..E-6, patrz `inputs/run_manifest_all.csv`) | wejście dla `an_early`; zmienna `EARLY_RESULTS` |
| Wyniki analiz (generowane) | `thesis_analysis/out/figures/*.pdf\|png`, `out/tables/*.tex`, `out/tables/numbers.tex` | kopiowane do `Masters-thesis-document/thesis/figures/generated/` i `thesis/tables/` |

## 2. Jak odtworzyć (z korzenia repo `Thesis - 2`)

```bash
cd thesis_analysis
# ścieżki domyślne wskazują na foldery obok (RESULTS, DATA, RESULTS_V2SHIFT); można nadpisać:
#   A100_RESULTS (domyślnie RESULTS/), DATA_DIR, QUICK_RESULTS, V2SHIFT_RESULTS, EARLY_RESULTS, O1_RESULTS, THESIS_OUT (folder wyjściowy)
./run_all.sh          # kolejno: an_runs an_v2_skill an_economics an_ga an_data an_v1 an_magnitude an_quick an_v2shift an_inventory an_early an_o1
```

Uwaga: `run_all.sh` na początku usuwa i tworzy od nowa domyślny folder `thesis_analysis/out`. Wymagania: Python 3.11 (środowisko, w którym powstały wyniki), numpy, pandas, matplotlib, scipy, scikit-learn (wersje użyte do wyników w pracy: pandas 2.3.3, numpy 2.3.4, matplotlib 3.11.2, scipy 1.17.1, scikit-learn 1.7.2). `an_data`, `an_v1`, `an_quick` i `an_v2shift` importują moduły potoku z korzenia repo (`data_processor*.py`, `feature_engineer*.py`), więc skrypty muszą leżeć w `Thesis - 2/thesis_analysis/`. Wszystkie bootstrapy i permutacje mają stałe ziarno; ponowne uruchomienie daje identyczne liczby i tabele (sprawdzone 2026-09-19).

Kolejność w pracy: tabele `.tex` są wczytywane przez `\input{tables/...}`, rysunki przez `\includegraphics{generated/...pdf}`, liczby w tekście to makra `\num...` z `tables/numbers.tex`. Aby znaleźć źródło liczby w tekście: `grep -rn "NazwaMakra" thesis_analysis/` (każde makro jest rejestrowane w skrypcie funkcją `register_number`).

## 3. Mapa: skrypt -> wyniki

| Skrypt | Wejście | Rysunki | Tabele |
|---|---|---|---|
| `an_runs.py` | `RESULTS/<przebieg>/` (`best_chromosome.json`, `metrics.json`, `test_trade_log.csv`) | F-10, F-11, F-15 | T-05, T-07, T-08, T-08b, T-10 |
| `an_v2_skill.py` | `RESULTS/<przebieg>/test_trade_log.csv` (macierze pomyłek i metryki liczone z logu prognoz) | F-12, F-13, F-13b, F-14, F-22 | T-07b, T-07c |
| `an_economics.py` | `RESULTS/<przebieg>/test_trade_log.csv` | F-18, F-19, F-20, F-21 | T-09, T-09b, T-09c |
| `an_ga.py` | `RESULTS/<przebieg>/` (`ga_population_history.jsonl`, `ga_history.csv`) | F-09, F-16, F-17, F-17b | T-11b |
| `an_data.py` | `DATA/*.csv` (surowe dane 5-min) + moduły repo `data_processor_v2.py`, `feature_engineer_v2.py` | F-03, F-04, F-05, F-06 | T-01, T-02, T-12, T-12b |
| `an_v1.py` | `DATA/*.csv` + `thesis_analysis/inputs/other_runs.csv` (23 wiersze: 18 przebiegów V1 i 5 wczesnych V2; wartości zebrane z `metrics.json` folderów w `RESULTS/` i sprawdzone z nimi 2026-09-20; skrypt sam **nie** czyta folderów `RESULTS/`) + moduły `data_processor.py`, `feature_engineer.py` | F-08 | T-06 |
| `an_magnitude.py` | `RESULTS/<przebieg>/test_trade_log.csv` | - | T-07d |
| `an_quick.py` | `RESULTS/` (szybkie przebiegi V1 z ziarnami, tworzone przez `quick_runs.py`) + `DATA/` + moduły V1 | F-23 | T-13 |
| `an_inventory.py` | `inputs/run_manifest_all.csv` (foldery nie są czytane) | - | T-17, T-18 |
| `an_early.py` | `RESULTS/<wczesny przebieg>/` (`metrics.json`, `config.json`, `ga_history.csv`, `ga_population_history.jsonl`, `test_trade_log.csv` dla E-6) + `RESULTS/<przebieg>/ga_history.csv`, `ga_population_history.jsonl` (osiem pełnych przebiegów) | F-26 | T-19 |
| `an_o1.py` | `RESULTS_O1/*.jsonl` (z `quick_o1_model2.py`; 20 osobników × 2 tryby × 4 warianty Model 2) | F-27 | T-20 |
| `an_v2shift.py` | `RESULTS_V2SHIFT/` (json + npz z `quick_v2_shift.py`) + `DATA/` + moduły V2 | F-24, F-25 | T-14, T-15, T-16 |

(F-xx i T-xx to numery robocze z `THESIS_PLAN.md`; w pracy rysunki i tabele mają numery automatyczne, a w tekście odwołują się przez etykiety z tabel poniżej.)

## 4. Rysunki generowane skryptami (26)

| Plik (`figures/generated/`) | Gdzie w pracy | Etykieta | Skrypt | Opis (z podpisu) |
|---|---|---|---|---|
| `f03_split_timeline.pdf` | Rozdz. 6 | `fig:split_timeline` | `an_data.py` | Chronological 60/20/20 split of the 30-minute data, shown for JPM |
| `f04_data_coverage.pdf` | Rozdz. 6 | `fig:data_coverage` | `an_data.py` | Coverage of the five-minute data |
| `f05_stylised_facts.pdf` | Rozdz. 6 | `fig:stylised_facts` | `an_data.py` | Stylised facts of the five-minute regular-session returns of the 19 stocks |
| `f06_labels.pdf` | Rozdz. 6 | `fig:labels` | `an_data.py` | Class labels of the V2 pipeline |
| `f08_v1_results.pdf` | Rozdz. 7 | `fig:v1_results` | `an_v1.py` | Test-set results of the 18 runs of the first pipeline (see Table) |
| `f09_ga_convergence.pdf` | Rozdz. 7 | `fig:ga_convergence` | `an_ga.py` | Progress of the eight GA runs |
| `f10_sharpe_gap.pdf` | Rozdz. 7 | `fig:sharpe_gap` | `an_runs.py` | Sharpe ratio of the best individual of each run as seen by the GA (calibration set) and as reported for the same individual on the test set |
| `f11_approved_composition.pdf` | Rozdz. 7 | `fig:approved` | `an_runs.py` | Composition of the bars approved by Model 2 in the eight runs, by the class predicted by Model 1 |
| `f12_model2_confidence.pdf` | Rozdz. 7 | `fig:m2_confidence` | `an_v2_skill.py` | Model 2 on the test sets of the eight runs, pooled |
| `f13_confusion_pooled.pdf` | Rozdz. 7 | `fig:confusion` | `an_v2_skill.py` | Confusion matrix of Model 1 on the test sets, pooled over the eight runs and normalised by the actual class (rows) |
| `f13b_confusion_grid.pdf` | Załącznik A | `fig:confusion_grid` | `an_v2_skill.py` | Confusion matrices of Model 1 on the test set for the eight runs, in percent of the actual class (D = DOWN, N = NEUTRAL, U = UP) |
| `f14_class_shares.pdf` | Rozdz. 7 | `fig:class_shares` | `an_v2_skill.py` | Class shares of the test set (solid bars) and of the predictions of Model 1 (hatched bars) for the eight runs |
| `f15_feature_stability.pdf` | Rozdz. 7 | `fig:feature_stability` | `an_runs.py` | Stability of the feature selection of the best chromosomes of the eight runs |
| `f16_gene_evolution.pdf` | Rozdz. 7 | `fig:gene_evolution` | `an_ga.py` | Composition of the population by value of the threshold multiplier m (left) and the confidence threshold (right) over the generations, pooled over the eight runs (groups ) |
| `f17_ga_landscape.pdf` | Rozdz. 7 | `fig:landscape` | `an_ga.py` | Calibration Sharpe ratios of all valid individuals of the eight runs |
| `f17b_gene_effects.pdf` | Rozdz. 7 | `fig:gene_effects` | `an_ga.py` | Mean calibration Sharpe ratio of the valid individuals by value of each hyperparameter gene, relative to the run mean (whiskers) |
| `f18_equity_curves.pdf` | Rozdz. 7 | `fig:equity` | `an_economics.py` | Cumulative profit of an equal-weight portfolio of the 19 stocks that follows the directional calls of Model 1 of each run |
| `f19_breakeven.pdf` | Rozdz. 7 | `fig:breakeven` | `an_economics.py` | Break-even one-way cost of the directional calls of each run (bars) with the 95% day-bootstrap interval (whiskers) |
| `f20_uncertainty.pdf` | Rozdz. 7 | `fig:uncertainty` | `an_economics.py` | Uncertainty of the gross profit per call |
| `f21_edge_by_group.pdf` | Rozdz. 7 | `fig:edge_groups` | `an_economics.py` | Gross profit per call pooled over the eight runs (a) by stock and (b) by hour of the day (US Eastern time), with 95% day-bootstrap intervals |
| `f22_prediction_agreement.pdf` | Rozdz. 7 | `fig:agreement` | `an_v2_skill.py` | (a) Agreement (Cohen's kappa) between the class predictions of the seven 30-minute runs on the same test bars |
| `f23_quick_runs.pdf` | Rozdz. 7 | `fig:quick_runs` | `an_quick.py` | Seeded repeats of the first pipeline on JPM |
| `f24_v2_window_shift.pdf` | Rozdz. 7 | `fig:v2_window_shift` | `an_v2shift.py` | Effect of the window offset on the test set, one line per seed |
| `f26_early_fitness.pdf` | Rozdz. 7 | `fig:early_fitness` | `an_early.py` | Fitness in the GA logs of the early runs and of the eight complete runs |
| `f27_o1_fitness.pdf` | Rozdz. 7 | `fig:o1_fitness` | `an_o1.py` | Fitness of 20 random individuals with four variants of Model 2, in-sample (calibration set) and on the test set, in the fast and in the full mode |
| `f25_v2_shift_sessions.pdf` | Rozdz. 7 | `fig:v2_shift_sessions` | `an_v2shift.py` | Sign accuracy (a) and mean gross profit per call (b) by session of the decision bar (mean over three seeds) for the base window, the shifted window and the model-free rul |

## 5. Tabele generowane skryptami (25)

| Plik (`tables/`) | Gdzie w pracy | Etykieta | Skrypt | Opis (z podpisu) |
|---|---|---|---|---|
| `t01_data_inventory.tex` | Rozdz. 6 | `tab:data_inventory` | `an_data.py` | The 19 stocks used in the experiments |
| `t02_splits.tex` | Rozdz. 6 | `tab:splits` | `an_data.py` | Chronological 60/20/20 split of each stock, as used by the V2 pipeline, after resampling, computation of the 28 features (rows with undefined features are dropped) and la |
| `t05_inventory.tex` | Rozdz. 6 | `tab:inventory` | `an_runs.py` | Inventory of the eight completed GA runs (population 20, 50 epochs per individual) |
| `t06_v1_results.tex` | Rozdz. 7 | `tab:v1_results` | `an_v1.py` | Version 1 (regression on 5-minute returns, test set = last 10% of each series) |
| `t07_reported_vs_real.tex` | Rozdz. 7 | `tab:reported_vs_real` | `an_runs.py` | Metrics reported by the pipeline versus what the approved trades actually were |
| `t07b_v2_skill.tex` | Rozdz. 7 | `tab:v2_skill` | `an_v2_skill.py` | Model 1 (three-class) on the test set |
| `t07c_m2_discrimination.tex` | Rozdz. 7 | `tab:m2_discrimination` | `an_v2_skill.py` | Model 2 as a classifier of "Model 1 was correct" |
| `t07d_magnitude_direction.tex` | Rozdz. 7 | `tab:magnitude_direction` | `an_magnitude.py` | Information of Model 1 about the size and about the sign of the next move |
| `t08_chromosomes.tex` | Rozdz. 7 | `tab:chromosomes` | `an_runs.py` | Best chromosome of each GA run (hyper-parameter genes) |
| `t08b_selected_features.tex` | Załącznik A | `tab:selected_features_app` | `an_runs.py` | Features selected by the best chromosome of each GA run |
| `t09_economics.tex` | Rozdz. 7 | `tab:economics` | `an_economics.py` | Counterfactual gross P&L of Model 1's UP/DOWN calls, ignoring Model 2 (one bar holding period, basis points per call) |
| `t09b_sessions.tex` | Rozdz. 7 | `tab:sessions` | `an_economics.py` | Gross P&L of Model 1's UP/DOWN calls by trading session, pooled over the eight runs (regular session = 09:30-16:00 US Eastern time) |
| `t09c_net.tex` | Rozdz. 7 | `tab:economics_net` | `an_economics.py` | Mean net P&L per call (bps) after a one-way cost of 1, 2 or 5 bps per unit of position change (an illustrative assumption), and annualised Sharpe ratios of the daily P&L  |
| `t10_feature_frequency.tex` | Załącznik A | `tab:feature_frequency` | `an_runs.py` | Selection frequency of each of the 28 candidate features in the best chromosomes of the eight GA runs |
| `t11b_ga_summary.tex` | Rozdz. 7 | `tab:ga_summary` | `an_ga.py` | Summary of the genetic-algorithm search per run |
| `t12_baselines.tex` | Rozdz. 6 | `tab:baselines` | `an_data.py` | Reference points for the three-class task on the test set of run R30-1 (n=16,601) |
| `t12b_directional_baselines.tex` | Rozdz. 6 | `tab:directional_baselines` | `an_data.py` | Directional reference predictors on the test rows of run R30-1 (gross of costs) |
| `t13_quick_runs.tex` | Rozdz. 7 | `tab:quick_runs` | `an_quick.py` | Seeded repeats (seeds 1--3) of the V1 experiments |
| `t14_v2_window_shift.tex` | Rozdz. 7 | `tab:v2_window_shift` | `an_v2shift.py` | Effect of the one-bar window offset of the V2 pipeline on Model 1 (mean standard deviation over seeds 1--3) |
| `t15_v2_shift_sessions.tex` | Rozdz. 7 | `tab:v2_shift_sessions` | `an_v2shift.py` | Directional calls of Model 1 by session of the decision bar (30-minute bars, m = 0.3, mean over the seeds) |
| `t17_experiments_overview.tex` | Rozdz. 6 | `tab:experiments_overview` | `an_inventory.py` | Overview of all experiments of the thesis |
| `t18_inventory_all.tex` | Załącznik A | `tab:inventory_all` | `an_inventory.py` | Inventory of all runs of the thesis (folder names) |
| `t19_early_runs.tex` | Rozdz. 7 | `tab:early_runs` | `an_early.py` | Test-set metrics printed by the pipeline at the end of the early runs |
| `t20_o1_model2.tex` | Rozdz. 7 | `tab:o1_model2` | `an_o1.py` | Test O1: fitness of a random generation 0 when Model 2 is fitted with four variants of the classifier |
| `t16_rule_segments.tex` | Rozdz. 7 | `tab:rule_segments` | `an_v2shift.py` | Sign accuracy (%) of the model-free rule that takes the opposite of the sign of the last bar's return, on bars with a non-zero return and a non-zero previous return, on t |

## 6. Elementy NIE generowane skryptem analiz

Od 2026-09-21 **każdy** rysunek i każda tabela ma w pracy wiersz „Source: ...” pod spodem oraz wpis „Source: ...” w spisie rysunków / tabel (skrócony podpis `\caption[krótki tytuł. Source: ...]{...}`). Makra źródeł są w `main.tex` (`\srcdraw`, `\srcown`, `\srcexp`, `\srcdata` oraz formy długie z końcówką `L`); tekst źródła zmienia się w jednym miejscu. Tabele generowane skryptami dostają podpis i wiersz źródła z `write_table` na podstawie słownika `FLOAT_META` w `common.py` (klucz: etykieta LaTeX; wartość: krótki tytuł, makro źródła, skrypt). Wpis w `FLOAT_META` jest wymagany dla każdej nowej tabeli.

| Element | Gdzie | Pochodzenie |
|---|---|---|
| Rys. `fig:lstm_cell` (komórka LSTM) | Rozdz. 2, `thesis/figures/tikz/lstm_cell.tex` | **własny rysunek TikZ, narysowany od zera 2026-09-21 według równań (2.6)-(2.11)**; zastąpił obraz `fig_2_2_lstm_cell.png` o niepewnym źródle (plik nieużywany) |
| Rys. `fig:bilstm` (BiLSTM rozwinięty w czasie) | Rozdz. 2, `thesis/figures/tikz/bilstm.tex` | **własny rysunek TikZ, narysowany od zera 2026-09-21**; zastąpił `fig_2_3_bilstm.jpg` o niepewnym źródle (plik nieużywany) |
| Rys. `fig:ga_flowchart` (pętla GA) | Rozdz. 2, `thesis/figures/tikz/ga_flowchart.tex` | własny rysunek TikZ (2026-09-21); zastąpił własny obraz z draw.io `fig_2_5_ga_flowchart.png` (ciemne pola bez strzałek); plik draw.io `Masters-thesis-document/draw_io/Genetic algorithm flowchart.drawio` zostaje |
| Rys. `fig:taxonomy` (taksonomia metod) | Rozdz. 2 | własny, TikZ w `02_state_of_the_art.tex` (zastąpił `fig_2_1_taxonomy.png`, nieużywany; źródło draw.io: `draw_io/Taxonomy of forecasting approaches.drawio`) |
| Rys. `fig:pipeline`, `fig:chromosome` | Rozdz. 4 | własne, TikZ w `04_methodology.tex` |
| Tabele `tab:features`, `tab:ga_space` | Rozdz. 4 | wpisane ręcznie w `04_methodology.tex` na podstawie kodu (`feature_engineer_v2.py`, `ga_optimizer_v2.py`) |
| Tabele `tab:modules`, `tab:versions`, `tab:config` | Rozdz. 5, 6 | wpisane ręcznie na podstawie struktury repo, `requirements.txt` i konfiguracji przebiegów |
| Wzory | Rozdz. 3-4 | własny zapis; źródła literaturowe w tekście |

Obrazy `fig_2_1_taxonomy.png`, `fig_2_2_lstm_cell.png`, `fig_2_3_bilstm.jpg`, `fig_2_5_ga_flowchart.png` w `thesis/figures/` nie są już używane w pracy.

## 7. Do zrobienia w kwestii źródeł

- ~~Potwierdzić pochodzenie obrazów z rozdz. 2~~ - zrobione 2026-09-21: wszystkie rysunki rozdz. 2 są własnymi rysunkami TikZ.
- ~~Podpisy „Źródło” pod rysunkami/tabelami~~ - zrobione 2026-09-21 (patrz początek sekcji 6).
- Dane cenowe pochodzą z eksportu LSEG (dawniej Refinitiv); decyzja Mateusza, czy dodać zdanie o licencji / warunkach użycia danych (w toku).
- `RESULTS/` jest jedynym źródłem prawdy dla wyników (decyzja z 2026-09-20; ścieżka domyślna w `common.py` zmieniona z `A100-results/RESULTS` na `RESULTS`). Pozostałe w `A100-results/`: `checkpoints/` (pliki `.pkl`, `model.pt`) i `analysis/run_summary.csv`.
- Wyniki (`RESULTS/`) Mateusz spakuje sam do archiwum zip; zostają poza git (decyzja 2026-09-21). Kod w `thesis_analysis/`, `quick_runs.py`, `quick_v2_shift.py` i `quick_o1_model2.py` nie jest jeszcze w commicie git; commit + znacznik tylko na jego wyraźne polecenie.

## 8. Mapa oznaczeń przebiegów: praca <-> folder <-> checkpoint

Nazwy folderów z Colaba zawierają czas UTC, z laptopa - czas lokalny (+02:00). Numery `GA_RUN_x` w `A100-results/checkpoints` są przesunięte względem oznaczeń w pracy.

| W pracy | Folder w `RESULTS/` (A100) | Checkpoint |
|---|---|---|
| R30-0 | `ga_full_v2_multi_all_stocks_20260618_095350` | `GA_RUN_1` |
| R15 | `ga_full_v2_multi_15min_all_stocks_20260620_141559` | `GA_15_RUN_1` |
| R30-1 | `ga_full_v2_multi_30min_all_stocks_20260621_125311` | `GA_RUN_2` |
| R30-2 | `ga_full_v2_multi_30min_all_stocks_20260621_215451` | `GA_RUN_3` |
| R30-3 | `ga_full_v2_multi_30min_all_stocks_20260622_124903` | `GA_RUN_4` |
| R30-4 | `ga_full_v2_multi_30min_all_stocks_20260622_233819` | `GA_RUN_5` |
| R30-5 | `ga_full_v2_multi_30min_all_stocks_20260623_111437` | `GA_RUN_6` |
| R30-6 | `ga_full_v2_multi_30min_all_stocks_20260623_210817` | `ga_full_v2_multi_30min_checkpoint.pkl` (katalog główny) |

Podział 47 folderów w `RESULTS/`: 15 powtórzeń V1 z ziarnami (CPU), 18 przebiegów V1, 6 wczesnych V2 (E-1..E-6; E-5 = `ga_full_v2_multi_all_stocks_20260616_171052`, przerwany po 10 pokoleniach, zachowano tylko `ga_history.csv`), 8 pełnych GA. Osiem przerwanych startów bez wyników (tylko `config.json` i kilka linii logu) przeniesiono 2026-09-20 do `_do_usuniecia_przerwane_przebiegi/` (do ręcznego usunięcia).

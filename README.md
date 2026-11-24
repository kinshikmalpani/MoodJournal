# MoodJournal prototype

## GitHub checkout

The repository includes application and training code, tests, research documentation, attributed external evaluation data, recorded reports, and the trained TF-IDF baseline. Neural weight files and the full continuation checkpoint are packaged separately by `python package_release.py` as `transfer/MoodJournal-models-v1.zip`; extract that trusted bundle into this project root to restore transformer inference and training state. The bundle contains a checksum manifest. Availability of a hosted bundle is recorded in the GitHub Releases section when uploaded.

Personal journals, demo databases, credentials, virtual environments, caches, transfer backups, and rendering intermediates are excluded. The original `Emotions Dataset.zip` and unknown-source `data.zip` are also excluded because their redistribution provenance has not been verified. Training and split-reproduction tests require the original archive supplied separately; the existing baseline can run without it. The source checksum is recorded in the reports. The presentation guide DOCX is a draft whose visual layout verification was not completed.

After creating an environment and installing the dependencies below, run `python -m streamlit run app.py`. With only the repository contents, the available trained model is the baseline; extracting the model bundle enables the transformer comparisons and v2 default. These are local research artifacts, not a hosted web deployment. The model uses the official [DistilBERT base uncased checkpoint](https://huggingface.co/distilbert/distilbert-base-uncased); consult its upstream model card and licence as well as the attribution files for external research data.

A local, single-user English journaling research prototype comparing TF-IDF/logistic regression with partially fine-tuned DistilBERT. The final prototype selects DistilBERT v2 by validation macro-F1. The original `MoodJournalApp.py` and synopsis describe earlier plans; use `app.py` and this document for implemented scope. BiLSTM, PHQ scoring, Spotify integration, and clinical diagnosis are not implemented.

## Presentation quick start

The current workspace contains trained models and a Python environment. No retraining is needed:

```powershell
# Personal journal: http://127.0.0.1:8501
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run.ps1

# Separate presentation journal: http://127.0.0.1:8502
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run.ps1 -Demo
```

Run one command and keep its terminal open; Ctrl+C stops the server. Demo mode seeds seven fictional entries on first use in `local_data/demo.sqlite3` and preserves them on later launches. Your personal database is not copied or used. The older Mac transfer ZIP includes personal entries and is not a presentation bundle.

### Five-minute demonstration

1. Start demo mode and point out the fictional-data banner. Explain the six supported emotions.
2. Write an example and choose **Analyse & save**. Explain that the bars are model scores, not measured feelings.
3. Try two emotions in separate sentences. Expand the passages to explain the mixed summary.
4. Open **Timeline & history** to show calendar trends, persistence, export, and deletion. Delete a newly added demo entry.
5. Open **About the project** to compare the baseline and transformer, then switch models in the sidebar. Old predictions retain their model version.
6. Show a limitation: `I do not resent my brother. I adore him and treasure our closeness.` Negation and passage aggregation can produce misleading results. Neutral text is not reliably rejected.

## Final experiment results

All models use the same cleaned train/validation/test counts: **15,923 / 1,993 / 2,000**. Values below are percentages; macro-F1 weights each class equally.

| Model | Validation macro-F1 | Test accuracy | Test macro-F1 |
|---|---:|---:|---:|
| TF-IDF + logistic regression | 86.10 | 88.25 | 83.97 |
| DistilBERT, recovered first epoch | 82.41 | 84.35 | 78.54 |
| DistilBERT, three additional epochs | 88.83 | 90.30 | 86.36 |
| DistilBERT v2, three further epochs | **89.01** | **91.50** | **87.61** |

V2 starts from the best continued checkpoint and fine-tunes the final two encoder layers and classifier with fresh AdamW state, learning rate 1e-5, batch size 32, seed 42, 96-token inputs, and a warmup/decay schedule. Epoch three was selected by validation macro-F1. This is seven epochs along the checkpoint lineage (1 + 3 + 3), with optimizer resets between experiments, not an uninterrupted seven-epoch run. V2 saves best inference weights but not full resumable optimizer state.

V2 improves test accuracy by 1.20 percentage points (24 additional correct examples) and test macro-F1 by 1.24 points over the continued model. Validation improves only 0.18 points. These are single-seed, repeatedly inspected benchmark results; statistical significance and robustness across seeds have not been established.

On the 28 single-emotion AI-authored journal diagnostics, passage macro-F1 rises from 78.28% to 81.10%, while whole-entry macro-F1 falls from 96.63% to 85.56%. Both models detect both expected passage labels on 6/6 mixed cases, and neither abstains on any of six neutral cases. These small diagnostics do not establish general journal accuracy. V2 confuses some love examples with joy and mishandles negated fear/anger. The existing passage method is retained with these limitations rather than tuned against these examples.

The app selects the completed model with the highest validation macro-F1 directly from reports. Earlier models remain available. `artifacts/distilbert_experiment_2/report.json` records v2 benchmark metrics; `artifacts/model_comparison.json` contains journal diagnostics for every available model. Regenerate with `python evaluate_models.py`; this performs inference only and never reads personal entries.

### Research rationale and presentation claims

Research question: how does a compact pretrained transformer compare with a sparse linear baseline for six-class emotion classification, and what breaks when it is applied to journals?

The contribution is an evaluated application and comparison with duplicate controls, model-versioned history, passage diagnostics, and local inference. It is not a new neural architecture or a clinically validated screening tool. Macro-F1 helps expose weaker minority classes; accuracy alone is dominated by more common classes. Test data is excluded from training and model selection, but repeated evaluation means it is not an untouched external assessment.

Relevant primary literature:

- Sanh et al. (2019), [DistilBERT, a distilled version of BERT](https://arxiv.org/abs/1910.01108). Motivates using a smaller pretrained transformer under limited compute. The paper's performance numbers are not our project's measurements.
- Saravia et al. (2018), [CARER: Contextualized Affect Representations for Emotion Recognition](https://aclanthology.org/D18-1404/). Related work on contextual emotion representations. This citation alone does not authenticate the provenance or licence of the locally supplied archive.

The local archive is identified by its SHA-256 in evaluation reports. Do not claim a verified original download source. The separate unknown-source `data.zip` is excluded. Future research should prioritise consented, independently annotated journals, neutral/OOD handling, negation, calibration, and repeated seeds rather than further tuning on these diagnostics.

For a presentation: introduce the problem, contrast TF-IDF with contextual transfer learning, show the dataset cleaning and train/validation/test roles, present the results table, demonstrate the app, and close with observed failure cases and future work.

## Literature review and presentation structure

### Review scope

This is a focused narrative review of nine primary works, updated 12 November 2026, not a systematic review or an exhaustive survey. Sources were located through ACL Anthology, arXiv, PMLR, and the original JMIR article. Search topics included emotion-recognition generalisation, GoEmotions, confidence calibration, diary-text analysis, strong sparse-text baselines, and behavioural testing. Inclusion prioritised primary research directly relevant to classification, deployment constraints, annotation, transfer, and evaluation. General mental-health motivation alone does not justify our classifier or timeline. Article records/abstracts and selected full-text methods, results, and limitations were checked. The review covers foundational and selected later work through 2024, plus a 2025 correction; it is not a complete survey of 2025–2026 publications. Bibliographic records are in `research/references.bib`.

### Critical comparison matrix

| Paper | Method and evidence | Limitation for our use case | Connection to MoodJournal and status |
|---|---|---|---|
| Wang & Manning (2012), [Baselines and Bigrams](https://aclanthology.org/P12-2018/) | Studies sparse classifiers and unigram/bigram features on sentiment and topic tasks. | Sentiment/topic results are not six-class journal-emotion results. | Supports testing a strong inexpensive baseline. Our TF-IDF/logistic-regression model is implemented; it is not a reproduction of their NB-SVM method. |
| Saravia et al. (2018), [CARER](https://aclanthology.org/D18-1404/) | Uses graph-derived patterns enriched with embeddings for emotion representations. | Different method and evaluation setting; does not validate journal trajectories. | Related work on contextual emotion representation. CARER is not implemented; the citation does not authenticate our archive. |
| Sanh et al. (2019), [DistilBERT](https://arxiv.org/abs/1910.01108) | Distils a pretrained language model into a smaller transformer and evaluates downstream language tasks. | General language-task performance is not journal-emotion validation. | We use the pretrained architecture and partially fine-tune it. We did not perform knowledge distillation ourselves. |
| Demszky et al. (2020), [GoEmotions](https://aclanthology.org/2020.acl-main.372/) | Introduces human-labelled Reddit comments with 27 emotion categories or neutral and transfer experiments. | Reddit text differs from journals; taxonomy differs from our six labels. | Highlights annotation and neutral/multiple-emotion requirements. This dataset is not used in our current experiments; merging label taxonomies would need explicit justification. |
| Zanwar et al. (2022), [Improving the Generalizability of Text-Based Emotion Detection](https://aclanthology.org/2022.nlpcss-1.1/) | Combines transformers with BiLSTM psycholinguistic features; reports within-domain and cross-domain experiments. | Improvements depend on dataset and transfer setup; their hybrid system differs from ours. | Motivates separating benchmark evaluation from journal transfer. Our small diagnostics are implemented; independent target-domain evaluation and their hybrid architecture are not. |
| Guo et al. (2017), [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html) | Examines confidence calibration and post-processing temperature scaling on classification tasks. | Calibration in one domain does not establish calibrated journal scores or emotion intensity. | Supports treating softmax output as model scores. Calibration remains proposed; our threshold selection is not probability calibration or neutral detection. |
| Shin et al. (2024), [Using Large Language Models to Detect Depression From User-Generated Diary Text Data](https://www.jmir.org/2024/1/e54617/) | Studies 428 diaries from 91 participants, LLMs, and questionnaire-based assessment over a two-week writing period. | Different clinical target, limited sample and demographics; the article notes generalisability limitations. | Demonstrates the need for independent participant evidence. We do not reproduce this study, validate depression detection, or compare its accuracy with ours. Use the corrected article version linked by the publisher. |
| Ribeiro et al. (2020), [Beyond Accuracy: Behavioral Testing of NLP Models with CheckList](https://aclanthology.org/2020.acl-main.442/) | Defines minimum-functionality, invariance, and directional tests; exposes failures not captured by aggregate task scores. | Passing a finite behavioural suite is not proof of general robustness; its evaluated tasks differ from ours. | Provides a framework for extending our existing diagnostic cases into controlled negation, paraphrase, and context tests. The CheckList package and full methodology are not currently implemented. |
| Koufakou et al. (2024), [Towards a new Benchmark for Emotion Detection in NLP](https://aclanthology.org/2024.genbench-1.13/) | Harmonises emotion corpora and evaluates a distilled RoBERTa model, including repeated runs and cross-corpus experiments. | Corpus sampling, taxonomy mapping, and protocol differences complicate direct comparisons. | Supports explicit label mapping, independent-domain evaluation, and repeated seeds. We have not run this unified benchmark or established equivalence to its corpus versions. |

Source-check notes: DistilBERT §4/Tables 1–3 evaluate general language tasks, IMDb and SQuAD, not our journal application. CARER §4 uses a CNN with enriched patterns, not our transformer. GoEmotions §6.2–6.3 includes target-domain fine-tuning; its transfer results must not be described as evidence of universal zero-shot transfer. Zanwar et al. §6/Tables 1–3 separate transfer setups. Guo et al. §5 discusses calibration outcomes and exceptions. CheckList §2/Figure 1 explains test types. Koufakou et al. §4.1 uses repeated experiments and §4.2 cautions about comparisons across differing protocols. Shin et al.'s Methods and Limitations supply the study design and sample restrictions; the [2025 correction](https://www.jmir.org/2025/1/e79198) changes a funding grant number, not reported model results.

### Synthesis and research questions

Sparse classifiers supply an inexpensive reference point; compact contextual models offer a practical alternative. Emotion datasets expose differing label definitions, while cross-domain research shows that source-domain scores alone are insufficient. Calibration literature separates model confidence from correctness. Diary studies illustrate the additional evidence required before moving from textual predictions to claims about people. Together these motivate a bounded comparative study, not a claim that journaling analysis is new.

- RQ1: Does partially fine-tuned DistilBERT outperform TF-IDF/logistic regression on the same cleaned six-class benchmark? Measured; v2 is better in this single-seed comparison.
- RQ2: Does that advantage transfer to independently labelled journal text, including neutral, mixed, and negated expressions? Only preliminary AI-authored diagnostics exist; the main question remains open.
- RQ3: Can daily aggregation faithfully summarise predicted textual themes? Computation is implemented and tested; agreement with human daily annotations remains unmeasured.

Do not retrospectively claim that every paper drove the original implementation. The matrix distinguishes implemented choices, supporting rationale added during review, and proposed follow-up experiments.

Our six labels include love but not disgust; do not call them an implementation of Ekman's six-emotion taxonomy. They are the categories supported by the current dataset. Also, “contextualized” in CARER is not synonymous with transformer embeddings, and pretrained distillation is not the same operation as our task-specific fine-tuning.

### Independent journal pilot protocol (ready to plan, not executed)

**Purpose:** evaluate transfer to journal-style writing separately from performance on the source benchmark. This is a nonclinical text-classification pilot. It cannot establish benefit to wellbeing or substitute for clinical validation.

**Freeze before evaluation:** record exact model-weight checksums, tokenizer versions, inference settings, passage rules, threshold policy, label definitions, metrics, and exclusions. Include baseline, continued DistilBERT, and v2; identify baseline-versus-v2 as the primary comparison and the continued model as a secondary checkpoint analysis. Do not change the default based on the eventual test results without reporting a new development cycle.

**Collection:** a planning target of 20–30 adult volunteers contributing approximately 7 dated entries each would yield roughly 140–210 entries if complete. This is a feasibility target, not a power calculation or a guaranteed representative sample. Recruit only after supervisor/institutional requirements and consent are settled. Allow participants to skip sensitive details and withdraw under an agreed process. Use pseudonymous author IDs and relative day indices; keep identity/contact information outside the model dataset. Keep raw entries local and out of public Git repositories, slides, or external AI services. Obtain separate permission before displaying any participant quotation. Use fictional examples in the demo by default.

**Annotation:** two annotators independently mark emotions expressed in the text, blinded to model predictions and user self-reports. Permit a set drawn from anger/fear/joy/love/sadness/surprise, plus separate neutral, outside-taxonomy, and unclear flags. Neutral means no expressed emotion identifiable under the guide; uncertainty means the annotator cannot decide. Do not equate negated fear with joy without positive evidence. Distinguish the author's expressed emotion from a quoted or third person's emotion. Record paragraph/passage labels as well as entry labels. Pilot the guide on separate material, revise it, then freeze it before final annotation. Preserve both original annotations before adjudication; report raw agreement and an appropriate chance-adjusted measure with label prevalence.

**Primary endpoint:** paired macro-F1 difference, v2 minus baseline, for adjudicated single-emotion entries whose labels are supported by both models. Always report the number and fraction of entries in this subset, per-class support, and all exclusions. This is a conditional endpoint, not overall performance on all journals. Report whole-entry and passage-mean inference separately and specify the primary method before seeing results (current app passage mean).

**Other endpoints:** for mixed entries evaluate predicted passage-label sets against human sets with per-label precision/recall and exact-set agreement. Separately measure neutral/outside-taxonomy false emotional assignments and abstention. Report ambiguous cases rather than silently dropping them from the study description. A model unable to predict neutral is allowed to fail this part of the evaluation. Use participant-level resampling for paired uncertainty intervals; do not treat multiple entries from one author as independent participants. Sparse classes and few authors limit inference. If tuning or calibration is introduced, reserve different authors for development and final evaluation.

**Timeline endpoint:** first verify software arithmetic using known artificial inputs (already partly covered by unit tests). Then aggregate human passage annotations and model passage assignments using the same entry/day weighting, and compare daily per-class frequencies and adjacent-day change directions. Define a tie/deadband rule on development material before testing; report observed-day counts and missingness. Do not infer emotion intensity from categorical frequencies. A separate optional daily self-report can examine association with experienced mood, but must not be mixed into textual ground truth. Inspect within-person patterns; a pooled correlation alone can be driven by differences between participants. Without real dated sequences, report no human trend-validation result.

**Acceptance and reporting:** run and disclose all frozen comparisons, including negative findings. A favourable pilot estimate supports a limited pilot claim, not guaranteed population superiority. If uncertainty is wide or the baseline wins, present that honestly and analyse failure patterns. Never manufacture human labels or present AI-generated fixtures as participant data.

### Changes with the strongest research value (proposed, not completed)

1. Freeze models and the evaluation plan, then collect an independent journal pilot with consent and two annotators blinded to model outputs. Annotate expressed emotions, allow neutral/mixed/unclear, report agreement and per-class support. Split by author if any development or tuning is performed. Do not force neutral entries into the six trained labels. Participant recruitment and institutional requirements must be resolved before collection.
2. Evaluate baseline, continued model, and v2 on identical entries. Compare whole-entry and passage methods separately. Report macro-F1 for the compatible single-label subset, neutral false-positive/abstention behaviour, and mixed-label precision/recall separately. Add paired uncertainty estimates; if authors contribute multiple entries, account for that clustering. Do not tune on the final test set.
3. Add a separate optional user self-report field and an explicit “not represented / unsure” option. Keep these distinct from predictions and independently annotated text labels. They can support a later longitudinal study; self-report is not automatically the ground truth of expressed textual emotion.
4. Make the timeline easier to interpret: display entry counts and an option for the fraction of passages assigned to each class. Label it as textual prediction frequency, not intensity or a validated mood measure. Use dated, human-annotated sequences for trend validation; fictional sequences test mechanics only.
5. Audit confidence and negation before further training. Temperature scaling can be evaluated using validation data and a separate test, but cannot create a neutral class or guarantee out-of-domain reliability. Keep the unvalidated three-day reflection heuristic out of headline research claims.

### Recommended twelve-slide storyline

| Slide | Question answered | Evidence or visual |
|---|---|---|
| 1. Title and scope | What is this project? | “MoodJournal: Evaluating a Compact Emotion Classifier for Local Journal Reflection.” |
| 2. User problem | Why organise journal writing? | A fictional week of entries; reviewing themes is the proposed use case, not an established clinical benefit. |
| 3. Why a dedicated app? | Why not only a conversation with AI? | Local execution, dated records, automatic charts, export, and fixed model versions. No claim that other AI cannot provide them. |
| 4. Literature synthesis | What do previous studies establish? | Group the matrix into baselines, contextual models, domain transfer, and validation. |
| 5. Research questions | What are we testing? | RQ1–RQ3 and their completed/pending status. |
| 6. Data and protocol | Is the comparison fair? | Archive checksum, duplicate controls, class counts, shared splits, validation selection. |
| 7. Architecture | How was it implemented? | Text → passages → local model → predictions → versioned SQLite records → calendar summaries. Passage output is inspection, not causal explanation. |
| 8. Benchmark results | Did contextual modelling help? | Baseline versus all transformer checkpoints; accuracy, macro-F1 and per-class errors. |
| 9. Transfer and failure analysis | Does better benchmark performance guarantee better journals? | Whole versus passage diagnostics, negation examples, neutral failures, AI-authored-set limitation. |
| 10. Working demonstration | Can the system run end to end? | Fictional demo: save → inspect passages → timeline → compare models. |
| 11. Evidence boundaries | What have we not proved? | Independent journal accuracy, real mood tracking, clinical effectiveness, repeated-seed robustness. |
| 12. Contribution and next experiment | What did we contribute and what comes next? | Controlled comparison + local workflow + honest error analysis; independent journal pilot. Include reference slides/appendix as needed. |

### Reconciliation with the original synopsis

The Phase I synopsis is a historical proposal, not the final experimental report. Its BiLSTM/GloVe architecture, clinical proxy, word-attribution module, Spotify features, and planned participant study must not be presented as completed. The diary study matching its description is Shin et al. (2024), in Journal of Medical Internet Research; verify the bibliography rather than repeating the synopsis's author/venue attribution. Remove unsupported generalisations such as “LLMs require cloud inference,” “most mood apps are stateless,” and “the study validates our hypothesis.” Journaling requires active user input. Averaging six emotion scores does not produce a validated PHQ-9 score. Any remaining clinical claims and other synopsis references require a separate source audit before inclusion in a final submission.

## External and synthetic evaluation (13 November 2026)

These are separate evaluation resources, not an independent human journal pilot. They were prepared without consulting model predictions, then frozen before inference. No personal journal database is used. None of these examples are used for training or default-model selection.

### External event descriptions

Source: Troiano, Padó and Klinger (2019), [Crowdsourcing and Validating Event-focused Emotion Corpora for German and English](https://aclanthology.org/P19-1391/), ACL. Download: [author-hosted archive](https://www.romanklinger.de/data-sets/deISEARenISEAR.zip). Original readme and ODC-By 1.0 database licence are retained in `research/pilot/source_readme.txt` and `source_license.txt`. Cite the authors when using the dataset or derived results.

The original enISEAR file contains 1,001 human-written descriptions elicited with emotion prompts, with reader-validation annotations. We retain the authors' `Prior_Emotion` prompt label as the target and preserve reader votes for inspection. These targets are not newly adjudicated journal labels. We retain the original masked emotion wording rather than insert the label into the input. Collection timestamps and event temporal-distance metadata are not journal dates. Demographic fields are omitted from the derived JSONL; source worker identifiers are retained solely as grouping keys for uncertainty estimates.

After removing two identical-text duplicates with matching labels, 999 descriptions remain. No normalized exact match was found against any of our original raw training, validation, or test texts. This does not rule out near-duplicates or pretrained-model exposure. The primary evaluation uses all 570 descriptions in the overlapping classes: anger 143, fear 143, joy 142, sadness 142. The remaining 429 descriptions concern disgust, guilt, or shame; they are retained and reported as unsupported-category cases. Love and surprise have no external targets here. We do not remap guilt or shame into sadness, or suppress love/surprise model outputs to make the task easier.

The primary endpoint is four-class macro-F1 of the app's mean passage scores. Accuracy and per-class precision/recall/F1 are also recorded. Whole-entry inference is a secondary comparison. Unsupported-category abstention is a separate diagnostic; it is not neutral detection. A paired 1,000-replicate bootstrap resamples source author groups for the baseline-v2 macro-F1 difference (seed 42), conditional on this selected corpus and endpoint. These are externally sourced event descriptions, not real dated journals or a representative population sample.

### Synthetic longitudinal stress test

`research/pilot/synthetic.jsonl` contains 32 AI-authored entries across four fictional people and 28 observed person-days. Two timelines span eight calendar days with one missing day; two span seven. Each person has a day with multiple entries. Scenarios include presentation anxiety and celebration, moving home, a workplace dispute, and ordinary social experiences. Passage-level intended labels, neutral passages, mixed emotions, negation, and missing days are explicit. There are six entirely neutral entries and eight mixed-emotion entries. The small designed set cannot estimate natural journal accuracy; model performance was not used to select or rewrite cases.

Trend diagnostics compare intended versus predicted passage-label frequencies: average passages within each entry, then give each entry equal weight within its day. Neutral or uncertain passages contribute zero to the six supported classes. We report daily frequency mean absolute error and direction agreement for calendar-adjacent days, separating changed from unchanged class frequencies. A fixed 0.05 deadband defines no change. Missing dates are never interpolated. This diagnostic frequency measure is distinct from the app's softmax-score chart and does not validate mood intensity or the three-day reflection heuristic.

### Measured results

External scores below use 570 supported-label event descriptions and macro-average over anger, fear, joy and sadness. They must not be compared numerically as if identical to the original six-class benchmark task.

| Model | External macro-F1: passage mean (primary) | External macro-F1: whole entry |
|---|---:|---:|
| TF-IDF baseline | 36.23% | 41.29% |
| Continued DistilBERT | **56.25%** | 56.84% |
| DistilBERT v2 | 54.32% | **59.29%** |

V2's primary improvement over baseline is 18.09 percentage points. The paired author-group bootstrap interval is +12.57 to +23.16 points (95% percentile interval; 1,000 replicates, 63 source-author groups). This supports an advantage on this external corpus and endpoint; it does not establish real-journal or clinical effectiveness. The prior continued checkpoint outperforms v2 with the app's passage averaging. Whole-entry results suggest context preservation may matter, but switching the app based on these results would be a new development decision requiring a fresh test, not an independently confirmed improvement.

| Synthetic diagnostic | Baseline | Continued | V2 |
|---|---:|---:|---:|
| Correct direction on 40 changed class transitions | 27.5% | 62.5% | 60.0% |
| Correct direction on 92 unchanged class transitions | 84.8% | 82.6% | 76.1% |
| Daily passage-frequency MAE, lower is better | 0.1220 | 0.1101 | 0.1250 |
| Entirely neutral entries returning uncertain | 2/6 | 1/6 | 0/6 |
| Mixed entries with exact expected emotion set | 0/8 | 2/8 | 2/8 |

The synthetic results are mixed: transformers recover more intended changes, but v2 has worse daily frequency error than the baseline and more false changes on unchanged class frequencies. Six class transitions from the same day pair are related observations, not independent participants. These small designed examples establish neither statistical superiority nor actual mood tracking. No models, thresholds, app defaults, or data were modified after observing these results. Three dedicated evaluation tests passed.

On the 429 unsupported-label descriptions (disgust/guilt/shame), the baseline abstained on 160, continued DistilBERT on 11, and v2 on 5. Abstention is not recognition of those categories, but this exposes how frequently the models force unsupported emotions into their trained classes.

### Reproduction commands and artifacts

- `research/prepare_pilot.py`: prepares data, attribution, exclusions, counts and model hashes; refuses to overwrite the frozen folder.
- `research/pilot/protocol.json`: frozen endpoints, dataset hashes, and model-weight hashes.
- `research/evaluate_pilot.py`: evaluates baseline, continued DistilBERT, and v2 without retraining. It checks hashes and records evaluator/runtime/tokenizer/configuration hashes before inference.
- `artifacts/pilot_evaluation/summary.json`: consolidated results and paired bootstrap interval.
- `artifacts/pilot_evaluation/<model>.json`: per-example predictions, class metrics, and synthetic daily frequencies, including failures.
- `tests/test_pilot.py`: checks calendar gaps, equal entry weighting, change-direction calculations, and intended/runtime passage alignment.

To rerun the frozen evaluation: `.\.venv\Scripts\python.exe research\evaluate_pilot.py`. Preserve the existing source archive, pilot folder, model artifacts and evaluation freeze. Changing data or methods requires a separately identified experiment. No new human annotation, participant recruitment, clinical study, or genuine longitudinal user validation was conducted.

## Run on Windows

From this folder in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-neural-lock.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Keep the `artifacts` folder when transferring: dependencies do not include trained weights. If no models exist, `python train_baseline.py` creates the baseline. After setup, only the final command is needed. Open http://127.0.0.1:8501. The model runs on CPU without external inference services. Model artifacts are trusted local pickle/joblib files: do not replace them with untrusted downloads.

## Training and evaluation

Training reads `Emotions Dataset.zip` directly. It normalizes case and whitespace for duplicate detection, excludes within-split conflicting groups, and removes cross-split duplicates from training/validation with test precedence. This is a cleaned evaluation protocol, not exactly the original published benchmark split. Test labels are not used to select model settings or thresholds.

Two logistic-regression candidates (ordinary and class-balanced) are compared by validation macro-F1. Vocabulary and TF-IDF statistics are fitted only on training. The selected model is evaluated on test without refitting. `artifacts/baseline_report.json` records data counts, archive checksum, validation trials, abstention threshold selection, classification metrics and confusion matrix. `artifacts/baseline.joblib` contains the model, label mapping, threshold and version.

The unknown-source `data.zip` is not used. No independent journal evaluation has yet been conducted. Short-text benchmark metrics do not establish journal accuracy or clinical validity.

## App behavior and limitations

- Entries are split at sentence/newline boundaries and into chunks of at most 80 words. All passages are retained. The entry summary averages passage predictions equally; this aggregation is experimental.
- Low-score and entirely out-of-vocabulary inputs can return `uncertain`. This is not a neutral detector. Model scores are uncalibrated and not emotion intensities.
- Multiple entries per day are retained. Daily charts average entries equally, use actual calendar windows, and leave missing days unobserved. Dates use Asia/Kolkata.
- Reflection prompts require three consecutive days ending today with daily negative-class score sum at least 0.6. This is an unvalidated heuristic, not an early-warning or diagnostic system.
- Entries persist in `local_data/journal.sqlite3` in plaintext. The folder may be synced by OneDrive because of its location. Export and individual deletion are available; exports and external backups are not deleted by the app.
- The app binds to localhost, has no authentication or multi-user separation, and disables Streamlit usage telemetry. It does not provide clinical scores or Spotify integration.

## Verify

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## DistilBERT experiment

### Resumable continuation

The new continuation is isolated in `artifacts/distilbert_continued`; the first-epoch model and report are preserved. Run `./continue_training.ps1` in PowerShell to start or resume it. It automatically evaluates the selected checkpoint, rebuilds the comparison, and runs verification after training completes. Logs from the launched background run are in `local_data/continuation.log` and `local_data/continuation-error.log`.

The experiment starts from the saved first-epoch weights with a **fresh optimizer and scheduler** because the original states were lost. It trains the same last two layers and classification head on the same cleaned training data. Maximum: three additional epochs. Early stopping: one epoch without a validation macro-F1 increase greater than 0.002. The highest validation score, including the starting checkpoint, determines the final model. Test metrics are computed only after selection; since this benchmark was already reported, these are repeated benchmark results rather than an untouched external evaluation.

Full checkpoints are atomically replaced every 50 training steps and before/after epoch validation. They contain model weights, AdamW state, learning-rate scheduler state, Python/NumPy/PyTorch RNG states, shuffled batch order, next batch, elapsed time, and validation history. The source weights and dataset are checksum-verified on resume. Only one process can train this experiment at a time. The tests verify that restored training produces an identical next optimizer update on CPU. Exact reproducibility assumes the same software and hardware environment; loading a trusted checkpoint on a different machine need not produce bitwise-identical results.

To resume directly: `.\.venv\Scripts\python.exe continue_transformer.py --resume`. Do not load a `training_checkpoint.pt` received from an untrusted source. No personal journals or development fixtures are used for training.

The archived recovered model is the completed **first-epoch checkpoint**. The original run stopped during epoch two before saving its later updates or final evaluation. `finalize_transformer.py --recovered-epoch 1` evaluates the surviving checkpoint without retraining. This is explicitly recorded in the model report and app; it is not a completed two-epoch result. The training script now persists epoch history and checkpoint provenance for future runs. The final prototype now uses v2; these instructions document the earlier experiment lineage.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-neural.txt
.\.venv\Scripts\python.exe train_transformer.py
.\.venv\Scripts\python.exe research/build_development_set.py
.\.venv\Scripts\python.exe evaluate_models.py
```

The transformer script downloads the official `distilbert/distilbert-base-uncased` checkpoint into a project-local cache. It uploads nothing and never reads `local_data`. It uses the same cleaned splits as the baseline, seed 42, two epochs, a 96-token training limit, AdamW, and validation macro-F1 checkpoint selection. Training batches group examples by token length and shuffle batch order to avoid wasted padding; validation/test retain their original order. Its final two encoder layers and classifier are trained; earlier layers are frozen to reduce CPU cost. `--trainable-layers 6` enables full fine-tuning for a separate future experiment. A completed model is protected against accidental overwriting. Exact installed dependencies are recorded in `requirements-neural-lock.txt` for the current Windows/Python environment.

Reports and progress are under `artifacts/`. The app only offers the transformer after metadata marks checkpoint evaluation complete. At runtime the model loads locally without downloads. Inference splits into token-overflow windows at 128 tokens so long input is retained. Training and inference lengths differ deliberately; the benchmark report uses the training limit and journal diagnostics use the runtime policy.

`research/journal_development.jsonl` contains 41 AI-authored development cases (28 single-emotion, 6 neutral, 6 mixed, 1 artificial length test). These labels have not been independently reviewed. They are diagnostics, not an independent journal benchmark, and are never used as training data. The comparison separates single-label macro-F1, neutral abstention, mixed-passage coverage, and warm-model CPU latency. It compares whole-entry inference with equal-weight passage aggregation. Neutral abstention is not validated neutral detection.

The default model is provisionally selected by original validation macro-F1, not by test results or these development examples. Research controls allow switching models. Existing saved predictions are not recomputed. Timeline charts filter to one model version; history/export retain every entry. A mixed summary is a passage-level heuristic, not a trained multilabel class.

References: [Hugging Face text classification workflow](https://huggingface.co/docs/transformers/tasks/sequence_classification), [official DistilBERT model card](https://huggingface.co/distilbert/distilbert-base-uncased).

Next research work: independent journal annotations, BiLSTM comparison, confidence calibration, repeated seeds, and a held-out journal evaluation. Keep final journal test data separate from model selection.

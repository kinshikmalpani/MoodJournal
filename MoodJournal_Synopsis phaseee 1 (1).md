**PROJECT SYNOPSIS**

> **Historical Phase I proposal:** This document describes planned work, not the final implementation or validated results. See README.md for the current DistilBERT/TF-IDF prototype, reviewed literature matrix, measured results, and presentation structure. The clinical proxy, BiLSTM/GloVe model, Spotify integration, word-attribution module, and participant study described below were not completed in the current prototype. Claims and references below should not be copied into the final report without checking the current review.

**MoodJournal**

*NLP-Based Mood Trajectory Analysis for Mental Health Monitoring*

| Department | Computational Intelligence — Minor Degree |
| :---- | :---- |
| **Project Type** | Applied NLP \+ Mental Health — Phase I Idea Submission |
| **Technology Stack** | BiLSTM · GloVe 200d · Spotify API · Streamlit · SQLite |
| **Academic Year** | 2025–2026 |
| **Submission Phase** | Phase I — Idea Submission & Literature Review |

# **Acknowledgement**

We express our sincere gratitude to our faculty guide and the Department of Computational Intelligence for their continued guidance, mentorship, and encouragement throughout this project proposal. Their insights have been instrumental in shaping the research direction and helping us identify the most impactful avenues of investigation.

We also thank the open-source NLP and machine learning community, whose publicly available tools, frameworks, and datasets make research of this kind accessible to student researchers. The Kaggle platform, TensorFlow ecosystem, Streamlit team, and Stanford NLP group (for GloVe embeddings) have all contributed foundational resources to this work. Special appreciation is extended to the researchers whose published work in digital phenotyping, computational psychiatry, and affective computing has directly informed our literature review and proposed methodology.

This document is submitted as part of the Minor Degree requirements in Computational Intelligence. All referenced works, datasets, and tools have been duly cited in accordance with academic standards.

# **Abstract**

MoodJournal is a proposed longitudinal mental health monitoring application designed to passively track users’ emotional well-being through free-form daily journaling, without requiring structured questionnaires or clinical input. The central motivation behind this project is the growing recognition that mood disorders such as depression and anxiety develop gradually, and that the linguistic patterns in a person’s writing encode rich, detectable signals of psychological state that can be leveraged for early intervention.

The proposed system’s core is a Bidirectional Long Short-Term Memory (BiLSTM) neural network trained with GloVe 200-dimensional word embeddings on publicly available emotion-labelled datasets. The model is planned to classify free-form journal entries into six emotional categories — anger, fear, joy, love, sadness, and surprise — and accumulate the resulting probability vectors in a local SQLite database for longitudinal analysis over 7–14 day windows. An early-warning alert mechanism is proposed to flag sustained negative emotion streaks exceeding three consecutive days.

Beyond classification, the proposed architecture includes a word-level attribution module using a masking-based prediction-delta strategy to highlight which tokens most influenced each emotional classification, promoting transparency. Integration with the Spotify Web API is planned to provide dynamic, audio-feature-driven playlist recommendations calibrated to the detected emotional state (valence and energy targeting), replacing the static, hardcoded recommendations seen in existing tools.

The primary research contribution of this project is an empirical investigation into whether NLP-derived emotion trajectories from free-form journaling correlate meaningfully with validated clinical instruments (PHQ-9 and GAD-7) — a question that remains underexplored in the existing digital phenotyping literature. A user study with 20–30 participants is planned for Phase II to test this hypothesis. This synopsis presents the concept, literature basis, objectives, and proposed methodology for Phase I review.

# **List of Tables**

| Table No. | Title | Chapter |
| :---- | :---- | :---- |
| Table 2.1 | Comparison of Related Works and Research Gaps | Chapter 2 |
| Table 4.1 | Proposed System Architecture — Layer Description | Chapter 4 |
| Table 4.2 | Proposed Technology Stack Summary | Chapter 4 |
| Table 4.3 | Proposed BiLSTM Model Configuration | Chapter 4 |
| Table 4.4 | Emotion–Audio Feature Mapping (Spotify Integration) | Chapter 4 |
| Table 4.5 | Proposed Implementation Timeline | Chapter 4 |

# **List of Figures**

| Figure No. | Title | Chapter |
| :---- | :---- | :---- |
| Figure 4.1 | Proposed MoodJournal System Architecture Diagram | Chapter 4 |
| Figure 4.2 | Proposed BiLSTM Model Architecture (Layer Stack) | Chapter 4 |
| Figure 4.3 | Planned Mood Trajectory Chart — 7-Day Rolling View | Chapter 4 |
| Figure 4.4 | Proposed Word Attribution Heatmap Output (Conceptual) | Chapter 4 |
| Figure 4.5 | Planned Streamlit UI Wireframe — Three-Tab Layout | Chapter 4 |
| Figure 4.6 | PHQ-9 Proxy Score Derivation Flow (Proposed) | Chapter 4 |

# **Chapter 1: Introduction**

## **1.1 Background**

Mental health has emerged as one of the most pressing global health challenges of the twenty-first century. According to the World Health Organization, nearly one billion people worldwide live with a mental health disorder, yet the majority lack access to timely, adequate, and affordable care. Depression and anxiety disorders — the most prevalent of these conditions — are particularly insidious in their onset. Unlike acute physical ailments, they develop gradually: their symptoms accumulate over days and weeks before reaching a clinically significant threshold, making both self-identification and professional detection extraordinarily difficult.

Traditional mental health monitoring relies on standardised self-report instruments such as the Patient Health Questionnaire-9 (PHQ-9) and the Generalised Anxiety Disorder-7 (GAD-7). These instruments, while clinically validated and widely used, carry several well-documented limitations. They require effortful, retrospective recall over a two-week reference period, are administered episodically (typically at clinical visits), and are susceptible to recall bias and social desirability effects. They capture a single snapshot rather than the dynamic, day-to-day fluctuations that define the lived experience of mood disorders. For populations in low- and middle-income settings or without regular access to mental health professionals, these barriers are even more pronounced.

The rapid proliferation of smartphones, wearables, and digital applications has created an unprecedented opportunity to address these limitations. The continuous streams of behavioural, linguistic, and physiological data generated by everyday digital interactions constitute a rich, largely untapped source of mental health signal — a paradigm known as digital phenotyping (Onnela & Rauch, 2016). Within this paradigm, Natural Language Processing (NLP) has emerged as a particularly powerful tool: language encodes emotional state with remarkable sensitivity, and free-form text such as journal entries provides an ecologically valid, low-burden mechanism for passive data collection.

## **1.2 Problem Statement**

Despite significant advances in NLP and digital health, three critical gaps remain inadequately addressed by existing tools and research:

* Delayed detection: Mental health deterioration often goes undetected until it becomes severe, because the drift is gradual and self-awareness of mood changes is limited. Current tools provide no early-warning capability based on longitudinal language patterns.

* Burden of active recall: Clinical instruments like the PHQ-9 and GAD-7 rely on effortful, retrospective self-report, introducing recall bias, compliance failure, and questionnaire fatigue. A passive alternative that requires no structured input is needed.

* Statelessness of existing mood applications: Contemporary NLP-based mood applications classify the current entry in isolation but have no memory of prior emotional states. They cannot detect longitudinal patterns, trend reversals, or sustained deterioration — precisely the signals most relevant to mental health risk.

MoodJournal is proposed to address all three gaps by building a longitudinal, journal-driven pipeline that passively accumulates mental health signals from free-form writing, stores them over time, and analyses the resulting trajectory for clinically relevant patterns — without requiring a single structured questionnaire.

## **1.3 Motivation**

The motivation for this project is both technical and humanistic. From a technical standpoint, the convergence of high-accuracy deep learning classifiers (particularly BiLSTM networks trained on affective corpora), lightweight local storage solutions (SQLite), accessible deployment frameworks (Streamlit), and rich audio-feature APIs (Spotify) makes it feasible to build a sophisticated, privacy-preserving longitudinal monitoring tool that runs entirely on a standard consumer laptop without cloud inference.

From a humanistic standpoint, the project is motivated by the recognition that the gap between mental health need and mental health care is not primarily a knowledge gap but an access gap. Tools that operate passively, require no clinical expertise, and integrate into existing daily behaviours (such as journaling) have the potential to scale mental health monitoring to populations that currently have no alternative.

The literature consistently identifies two underexplored research opportunities that directly motivate this project: the use of journaling text (as opposed to social media data) as a source of mental health signal, and the empirical validation of NLP-derived trajectories against clinical instruments in a prospective study. MoodJournal is designed to address both.

## **1.4 Scope of the Project**

This Phase I submission covers the project’s conceptual foundation, theoretical justification, literature review, objectives, and proposed methodology and architecture. It represents the ideation and planning stage of the project.

Phase II (Final Submission) will cover the implementation outcomes, user study results, correlation analyses, and any publication-stage evaluation. The following components fall within the scope of this proposal:

* A BiLSTM \+ GloVe emotion classification pipeline for free-form journal text

* A longitudinal SQLite-backed journal store and trajectory analysis engine

* A word-level attribution module for prediction transparency

* Spotify audio-feature-based dynamic music recommendations

* A PHQ-9 proxy score derivation mechanism for research comparison

* A Streamlit web interface with three tabs for different user functions

* A planned empirical user study correlating NLP trajectories with PHQ-9/GAD-7

The project does not claim to be a clinical diagnostic tool. It is explicitly framed as a research companion and wellness aid. Deployment at scale, multi-user cloud architecture, and clinical validation trials are considered future scope and are discussed in Chapter 5\.

# **Chapter 2: Literature Review**

This chapter reviews the key bodies of research that underpin MoodJournal’s proposed design. The review spans NLP for mental health detection, recurrent architectures for emotion classification, word embedding methods, digital phenotyping, clinical instrument validation, music and emotion regulation, and emerging transformer-based approaches.

## **2.1 NLP for Mental Health Detection**

The application of NLP to mental health monitoring has grown substantially over the past decade, catalysed by the availability of large social media datasets and the rapid maturation of deep learning. Zhang et al. (2024) conducted a comprehensive narrative review of NLP applied to mental illness detection, finding that deep learning architectures — particularly recurrent and transformer-based networks — consistently outperform traditional machine learning approaches on emotion and mental health classification tasks. The annual CLPsych shared tasks at the Association for Computational Linguistics (2015–2024) have produced standardised benchmarks and annotated corpora for depression, suicide risk, and anxiety detection from text, forming the backbone of the field.

Coppersmith et al. (2014) were among the first to quantitatively demonstrate mental health signals in social media text, showing that individuals with diagnosed depression and PTSD exhibit measurable, statistically significant differences in language use compared to matched controls. Their work established passive, language-based monitoring as a scientifically credible paradigm. Subsequent studies have progressively refined these methods, with researchers showing that language features such as affect density, first-person pronoun usage, future-tense ratio, and negation frequency carry predictive information about psychological state.

A critical challenge identified across this literature is domain shift: models trained predominantly on social media text (typically short, informal, syntactically sparse sentences such as tweets) perform measurably differently on journaling text, which is longer, more reflective, more syntactically complex, and more contextually rich. This domain gap has received limited empirical investigation. MoodJournal’s proposed in-domain evaluation phase directly addresses this gap by collecting journal entries from consenting volunteers and quantifying performance degradation relative to the original social media test set.

## **2.2 Bidirectional LSTM for Emotion Classification**

Long Short-Term Memory (LSTM) networks, introduced by Hochreiter & Schmidhuber (1997), and their bidirectional variant (BiLSTM) have been extensively validated for sequential text classification. The BiLSTM processes input sequences simultaneously in forward and reverse temporal directions, enabling the model to capture dependencies across the full context of an entry — a critical capability for journal text, where emotional cues may appear anywhere in a passage and require global context for correct interpretation.

Luo et al. (2018) proposed the Self-Attentive BiLSTM (SA-BiLSTM) for emotion detection in dialogues, demonstrating that attending to contextual dependencies between utterances significantly improves the classification of ambiguous or blended emotional states. Pimpalkar and Raj (2022) specifically validated the combination of GloVe embeddings with a Multi-layer BiLSTM (MBiLSTM), achieving 92.05% test accuracy on the IMDB sentiment dataset, establishing GloVe-BiLSTM as a strong, reproducible baseline for affective computing tasks.

Kerz et al. (2023, Frontiers in Psychiatry) trained BiLSTM models on a comprehensive set of interpretable linguistic features — syntactic complexity, lexical sophistication, readability, cohesion, and sentiment — for automated mental health detection. Their explainability-focused approach, which prioritises the interpretability of features alongside predictive accuracy, aligns directly with MoodJournal’s proposed word-attribution module. The DABLNet architecture (Scientific Reports, 2025), which integrates textual and temporal features from Reddit posts using a deep attention BiLSTM, further validates the value of combining linguistic content with behavioural timing patterns for early mental health crisis detection — a philosophy central to MoodJournal’s longitudinal design.

## **2.3 Word Embeddings: GloVe and Alternatives**

Pennington et al. (2014) introduced GloVe (Global Vectors for Word Representation), a method that constructs dense word vector representations by factorising a global word co-occurrence matrix derived from a large corpus. Unlike Word2Vec, which relies on local context windows, GloVe captures both local and global statistical information, producing embeddings that excel at semantic and syntactic analogical reasoning. The 6B token, 200-dimensional variant (GloVe 6B 200d) — trained on Wikipedia 2014 and Gigaword 5 corpora — has become a widely used baseline for NLP classification tasks.

Studies systematically comparing GloVe with Word2Vec, FastText, and contextual embeddings (BERT, RoBERTa) on sentiment and emotion analysis tasks consistently find that GloVe 200d provides a strong balance of performance, interpretability, and computational efficiency. Critically for this project, GloVe embeddings do not require GPU inference infrastructure: a BiLSTM model with GloVe 200d embeddings can be trained and run on standard CPU hardware, making MoodJournal’s architecture viable as a standalone, locally-deployed application.

While transformer-based models such as BERT and its variants achieve state-of-the-art performance on many NLP benchmarks, they require substantially greater computational resources and are less interpretable. For a privacy-preserving, locally-run application where interpretability is a design requirement, the BiLSTM \+ GloVe combination represents a well-justified, literature-validated choice. The project’s future scope includes investigating whether fine-tuned transformer models (e.g., MentalBERT) would meaningfully improve performance in the journaling domain.

## **2.4 Digital Phenotyping and Longitudinal Monitoring**

Digital phenotyping — the moment-by-moment quantification of individual-level human phenotype using data from personal digital devices — was formally defined by Onnela & Rauch (2016) and has since become one of the most actively investigated paradigms in computational psychiatry. The core premise is that the behavioural traces left by individuals as they interact with digital devices — GPS movement, communication patterns, screen time, text content — collectively encode rich, temporally dense information about psychological state that cannot be captured by episodic clinical assessments.

Li et al. (2023, Frontiers in Psychiatry) identified improving the clinical value of digital phenotyping as a priority research direction, noting that passive data collection from smartphones and applications can capture dynamic symptomatology between clinical visits. Brito & Vieira-Andrade (2025) reviewed digital phenotyping studies across mood disorders and schizophrenia, finding meaningful correlations between passive biomarkers and clinical measures, and identifying speech and text analysis via NLP as a new and underexplored research avenue — directly positioning MoodJournal’s approach as timely and scientifically motivated.

Kim et al. (2024, JMIR Mental Health) specifically validated diary text as a mental health signal source: an emotional diary writing application paired with a large language model was used to detect depression in 91 participants, with PHQ-9 as the clinical ground truth. This study directly validates MoodJournal’s core hypothesis — that diary writing is a feasible and ecologically valid source of mental health signal — while also highlighting the gap this project aims to address: Kim et al.’s approach used an LLM (which requires cloud inference) and did not analyse longitudinal trajectories. MoodJournal proposes a lightweight, locally-executable, trajectory-aware alternative.

## **2.5 Clinical Instruments: PHQ-9 and GAD-7**

The Patient Health Questionnaire-9 (PHQ-9), developed by Kroenke et al. (2001), is a nine-item self-report instrument for assessing depression severity on a scale from 0 to 27\. The Generalised Anxiety Disorder-7 (GAD-7) is the analogous instrument for anxiety screening. Both have been extensively validated across diverse populations and clinical settings.

Suh et al. (2023, Psychiatry Investigation) confirmed that PHQ-9 and GAD-7 administered via smartphone applications retain equivalent psychometric properties to paper-based versions (Cronbach’s alpha \= 0.870 and 0.919 respectively), validating the use of digital administration in research contexts. Jiang et al. (2024, IEEE Journal of Biomedical and Health Informatics) demonstrated Pearson correlations between digital biomarkers and PHQ-9/GAD-7 scores ranging from 0.31 to 0.37 using facial expression and physiological measures, establishing a precedent for the level of correlation that passive digital monitoring methods can achieve with clinical instruments. MoodJournal’s proposed user study is designed to test whether NLP trajectory scores can achieve comparable or superior correlations using language data alone.

## **2.6 Music, Emotion Regulation, and Spotify Integration**

Music has a well-established role in emotion regulation. A 2022 meta-analysis published by the American Psychiatric Association found an overall beneficial effect of music interventions on stress-related outcomes across populations. Research on the functions of music in mood regulation (MMR) shows that individuals actively and strategically use music to modulate their emotional states — for distraction, emotional discharge, entrainment to a target mood, and social bonding.

The ‘Mood Music’ study (University of Bristol, IJPDS) combined Spotify streaming history, Ecological Momentary Assessment, and mental health survey measures (PHQ-9, GAD-7, SWEMWBS) from 171 university students, demonstrating that naturalistic music listening data can be used as a proxy for well-being. This study provides direct justification for MoodJournal’s planned Spotify integration: the audio features of music that individuals choose to listen to reflect and influence their emotional state. MoodJournal proposes to operationalise this relationship in reverse — by recommending playlists whose audio features (valence and energy, as defined by Spotify’s audio analysis API) are targeted to shift or support the detected emotional state.

## **2.7 Research Gaps Addressed by MoodJournal**

The following table synthesises the key research gaps identified in the literature and maps them to MoodJournal’s proposed contributions:

| Gap Identified | Existing Literature | MoodJournal’s Proposed Contribution |
| :---- | :---- | :---- |
| Stateless mood tools | Most NLP mood apps classify single entries without memory of prior states | Proposed SQLite longitudinal store accumulates daily emotion vectors; trajectory engine analyses trends over 7–14 days |
| Domain shift (social media → journaling) | Models trained on tweets/Reddit; limited journaling-domain evaluation exists | Planned in-domain evaluation on collected journal entries; quantification of performance degradation |
| Burden of active questionnaires | PHQ-9/GAD-7 require effortful, retrospective self-report | Free-form journaling as a passive signal source; PHQ-9 proxy score derived automatically from emotion averages |
| Opacity of deep learning classifiers | Most NLP mood tools are black boxes; predictions unexplained to users | Proposed word-masking attribution module highlights which tokens drove each prediction |
| Static music recommendations | Existing mood apps use hardcoded playlists or URI lists | Planned dynamic audio-feature-based Spotify recommendations using target\_valence and target\_energy |

# **Chapter 3: Objectives**

## **3.1 Primary Engineering Objectives**

The following engineering objectives define the system capabilities that MoodJournal aims to build and demonstrate through implementation:

* Design and implement a BiLSTM \+ GloVe 200d emotion classification pipeline trained on publicly available emotion-labelled text data, targeting competitive test accuracy on the held-out evaluation set.

* Construct a SQLite-backed journal store that accumulates daily emotion probability vectors per user across sessions, enabling persistent longitudinal tracking without cloud dependency.

* Develop a mood trajectory engine that applies a 7–14 day rolling analysis to stored emotion vectors, detects sustained negative emotion streaks, and triggers early-warning alerts with evidence-based coping suggestions.

* Implement a word-level attribution module using a masking-based prediction delta strategy to identify which tokens most strongly influenced each classification, enhancing transparency and user trust.

* Derive a PHQ-9 proxy score by mapping NLP emotion averages onto the clinical severity scale, enabling comparison with validated instruments in a research context.

* Integrate Spotify’s audio-feature recommendation API (target\_valence and target\_energy parameters) to generate dynamic, emotion-appropriate playlist recommendations.

* Build an accessible three-tab Streamlit web interface (Today’s Entry, Mood Timeline, About/Research) that requires no machine learning expertise from the user.

## **3.2 Research Objectives**

The following research questions will be investigated in the course of this project and evaluated empirically in Phase II:

* RQ1 — Accuracy in the Journaling Domain: How accurately does the BiLSTM classifier detect emotions from naturalistic journaling text, compared to its performance on the original social media training domain? What is the magnitude of the domain shift?

* RQ2 — Clinical Correlation: Does the NLP-derived emotion trajectory score, computed from 2–3 weeks of daily journaling, correlate significantly (Pearson r) with pre-to-post changes in PHQ-9 and GAD-7 scores in a prospective user study (N \= 20–30)?

* RQ3 — Linguistic Predictors: Which interpretable linguistic features — affect vocabulary density, future-tense ratio, first-person pronoun frequency, negation density — are most predictive of sustained negative mood trajectories when used as inputs to a logistic regression classifier?

## **3.3 System Design Objectives**

The following design constraints guide the technical architecture of MoodJournal:

* Privacy by design: all journal data and emotion vectors to be stored locally in SQLite; no data transmitted to external servers except Spotify API calls.

* Computational accessibility: the system is to run on standard consumer laptop hardware (CPU only) without cloud inference infrastructure.

* Non-clinical framing: the application to be explicitly framed as a research and wellness tool, not a clinical diagnostic instrument, in all user-facing text.

# **Chapter 4: Methodology**

## **4.1 Proposed System Architecture**

MoodJournal is designed as a four-layer sequential pipeline from raw text input to actionable output. The following table describes each proposed layer and its role in the system:

| Layer | Component | Proposed Description |
| :---- | :---- | :---- |
| Input | Journal Entry | Free-form daily text of any length, entered via Streamlit text area. No structured prompts or character limits. |
| NLP | Emotion Classifier | Proposed BiLSTM \+ GloVe 200d model producing 6-class softmax output (anger, fear, joy, love, sadness, surprise). Preprocessing: lowercasing, stopword removal, lemmatisation, URL removal, padding. |
| NLP | Attribution Engine | Word-masking strategy: each token blanked in turn; drop in top-emotion probability used as importance score. Results displayed as highlighted text in the UI. |
| Tracking | Longitudinal Store | SQLite database storing date, raw entry text, predicted label, and all six emotion probability scores per session. Accumulates across days and sessions. |
| Tracking | Trajectory Engine | 7–14 day rolling analysis of dominant emotion trends. 3+ consecutive days of dominant negative emotion triggers an early-warning alert with CBT-informed coping suggestions and support resource signposting. |
| Output | Spotify Integration | sp.recommendations() called with target\_valence and target\_energy parameters per emotion class. Dynamic recommendations derived from audio features, not hardcoded URIs. |

## **4.2 Proposed BiLSTM Model Architecture**

The emotion classifier is planned to follow a deep bidirectional LSTM architecture. The following table describes the proposed layer configuration:

| Layer | Proposed Configuration | Expected Output Shape |
| :---- | :---- | :---- |
| Embedding | GloVe 6B 200d · vocab\_size × 200 · non-trainable weights | batch × maxlen × 200 |
| BiLSTM \#1 | 64 units per direction · return\_sequences=True | batch × maxlen × 128 |
| BiLSTM \#2 | 64 units per direction · return\_sequences=False | batch × 128 |
| Dense | 64 units · ReLU activation · Dropout 0.3 | batch × 64 |
| Output | Softmax · 6 classes (anger, fear, joy, love, sadness, surprise) | batch × 6 |

Training is planned on the Kaggle Emotions NLP dataset (approximately 16,000 labelled sentences from the CARER paper by Saravia et al., 2018). SMOTE oversampling via the imbalanced-learn library will be applied to address class imbalance in the training set. The holdout evaluation set will be used to assess generalisation performance prior to the in-domain journaling evaluation.

## **4.3 Text Preprocessing Pipeline**

All journal entries will pass through a standardised preprocessing pipeline before inference:

* Lowercasing and Unicode normalisation

* URL and special character removal using regular expressions

* English stopword removal using the NLTK corpus

* WordNetLemmatizer for morphological normalisation (verb form)

* Tokenisation using the trained Keras tokenizer and sequence padding to a fixed maxlen determined from the training data distribution

## **4.4 Longitudinal Storage and Trajectory Engine**

Each analysed journal entry is planned to be stored in a local SQLite database with the following schema: date (TEXT), entry (TEXT), label (TEXT), and probability scores for each of the six emotion classes (REAL). The trajectory engine will apply a 7-day rolling window to compute the dominant emotion trend, calculate the proportion of negative emotion days (anger, fear, sadness), and trigger an early-warning alert if three or more consecutive days show a dominant negative emotion. A PHQ-9 proxy score is planned to be derived as a weighted average of sadness and fear probabilities, linearly mapped to the PHQ-9 scale (0–27) for research comparison purposes.

## **4.5 Word Attribution Module**

Transparency is planned to be achieved through a masking-based attribution strategy. For each inference, every token in the input sequence will be masked (replaced with a zero vector) one at a time. The resulting drop in the predicted probability of the top emotion class is recorded as that token’s importance score. Tokens will be highlighted in the Streamlit UI with colour intensity proportional to their importance score, allowing users to understand which words drove the classification and providing a basis for self-reflection.

## **4.6 Proposed Spotify Audio-Feature Recommendations**

MoodJournal plans to integrate Spotify’s Web API through the Spotipy Python library. Upon classification, the detected emotion will be mapped to a target valence (0 \= emotionally negative, 1 \= emotionally positive) and target energy (0 \= calm, 1 \= intense) following empirically grounded mappings derived from the circumplex model of affect. The sp.recommendations() endpoint will be called with these parameters, returning a dynamic, context-appropriate playlist rather than a static list of hardcoded tracks.

| Emotion | Target Valence | Target Energy | Musical Rationale |
| :---- | :---- | :---- | :---- |
| Anger | 0.30 | 0.80 | High-intensity, somewhat melancholic — cathartic emotional release |
| Fear | 0.20 | 0.40 | Low energy, dark — validating and accompanying the emotional state gently |
| Joy | 0.90 | 0.80 | Upbeat, high energy — reinforcing and sustaining positive affect |
| Love | 0.80 | 0.50 | Warm, moderate energy — emotionally rich and resonant |
| Sadness | 0.30 | 0.30 | Quiet, low energy — gentle accompaniment without forced positivity |
| Surprise | 0.70 | 0.70 | Lively, moderately positive — dynamic listening to match the state |

## **4.7 Proposed Technology Stack**

| Category | Technology | Purpose |
| :---- | :---- | :---- |
| Deep Learning | TensorFlow / Keras | BiLSTM model construction, training, and inference |
| Word Embeddings | GloVe 6B 200d | Pre-trained word vectors for the embedding layer |
| Data Handling | Pandas, NumPy | Dataset processing, feature manipulation, storage retrieval |
| NLP Utilities | NLTK, scikit-learn | Preprocessing, tokenisation, SMOTE, logistic regression (RQ3) |
| Local Storage | SQLite3 | Longitudinal journal and emotion vector persistence |
| Music API | Spotipy (Spotify Web API) | Audio-feature-based playlist recommendations |
| Frontend / UI | Streamlit | Three-tab web interface accessible to non-technical users |
| Visualisation | Matplotlib, Pandas | 7-day mood trajectory chart and emotion probability bar charts |

## **4.8 Research Methodology**

The publishable research contribution of this project rests on a three-phase empirical study planned for Phase II:

### **Phase 1 — In-Domain Evaluation**

The first research phase involves collecting 200 or more journal entries from consenting volunteers and evaluating the BiLSTM classifier on this in-domain data. The primary aim is to quantify the domain shift between the social media training domain and the journaling domain. Performance metrics (accuracy, macro-F1, per-class precision and recall) will be compared between the original held-out test set and the journaling evaluation set. This phase will also identify whether preprocessing adaptations are needed for the longer, more reflective text structure of journal entries.

### **Phase 2 — Longitudinal Correlation Study**

The second research phase involves recruiting 20–30 participants from the university community for a prospective 2–3 week longitudinal study. Participants will complete PHQ-9 and GAD-7 questionnaires at the start (T0) and end (T1) of the study period and will journal daily using the MoodJournal application. The NLP trajectory score (a weighted average of daily negative emotion probabilities over the study period) will be computed for each participant, and Pearson correlation between the trajectory score and ΔPHQ-9/ΔGAD-7 will be measured. Ethics: all data will be anonymised, PHQ-9 scores will not be shared across participants, and the system will be explicitly framed as a research tool.

### **Phase 3 — Linguistic Feature Analysis**

The third research phase uses logistic regression with interpretable linguistic features to identify the strongest predictors of a worsening mood trajectory. Four feature types are planned: affect vocabulary density (computed against LIWC-style affective dictionaries), future-tense ratio (proportion of future-tense verb forms), first-person pronoun frequency, and negation density. This follows the methodology of Kerz et al. (2023) and will identify which linguistic signals are most clinically informative.

## **4.9 Proposed Implementation Timeline**

| Phase | Duration | Key Planned Deliverables |
| :---- | :---- | :---- |
| 1 — Model & Core App | Weeks 1–3 | BiLSTM trained on Kaggle dataset, SQLite store implemented, trajectory engine, attribution module, Streamlit UI (3 tabs) |
| 2 — Integration | Weeks 4–6 | Spotify integration, PHQ-9 proxy score derivation, coping suggestions, early-warning alert system, UI/UX refinement |
| 3 — User Study | Weeks 7–9 | Recruit 20–30 university participants, 2-week daily journaling period, PHQ-9 \+ GAD-7 pre/post administration, Pearson correlation analysis |
| 4 — Analysis & Paper | Weeks 10–12 | Linguistic feature analysis (logistic regression), results write-up and visualisation, conference paper preparation, final project presentation |

# **Chapter 5: Conclusion**

MoodJournal represents a timely, technically grounded, and clinically motivated response to three well-documented gaps in the existing digital mental health monitoring literature: the statelessness of current NLP mood tools, the burden of active clinical questionnaires, and the opacity of deep learning classifiers. By proposing a longitudinal, locally-executing, transparency-first architecture built on BiLSTM emotion classification, SQLite-backed trajectory analysis, word-level attribution, and dynamic Spotify-based recommendations, this project aims to make passive mental health monitoring both scientifically rigorous and practically accessible.

The primary research contribution — an empirical investigation into whether NLP-derived emotion trajectories from free-form journaling correlate with validated clinical instruments (PHQ-9 and GAD-7) — is both novel and feasible within the project’s planned timeline. Comparable correlations (Pearson r \= 0.31–0.37) have been demonstrated using other passive digital biomarkers (Jiang et al., 2024), providing a reasonable benchmark. The literature on diary-based mental health detection (Kim et al., 2024\) directly validates the ecological validity of the data modality. If the proposed correlation study confirms a meaningful relationship between NLP trajectory and clinical instrument change, MoodJournal will provide a blueprint for questionnaire-free mental health monitoring that could scale to large populations.

## **5.1 Expected Impact**

If successfully implemented and validated, MoodJournal is expected to contribute in the following ways:

* A validated, open-source implementation of a longitudinal NLP-based mood monitoring pipeline that researchers can adapt and build upon.

* The first systematic comparison of BiLSTM emotion classification performance across social media and journaling text domains, quantifying the domain shift that has been assumed but not empirically measured.

* A prospective correlation study establishing (or challenging) the relationship between NLP trajectory scores and validated clinical screening instruments in a university population.

* A demonstration that interpretable, transparent, privacy-preserving mental health monitoring tools can be built and run on consumer hardware without cloud infrastructure.

## **5.2 Future Enhancement Scope**

The following enhancements are considered beyond the scope of Phase I and Phase II but represent natural extensions of this research:

* Transformer-based classification: Investigating whether domain-specific transformer models (e.g., MentalBERT, MentalRoBERTa) meaningfully improve emotion classification accuracy in the journaling domain beyond the GloVe-BiLSTM baseline.

* Multimodal expansion: Incorporating voice-note or audio diary entries using speech emotion recognition, enriching the signal with paralinguistic features (prosody, energy, speaking rate).

* Multi-user architecture: Extending the system to a cloud-based multi-user architecture with appropriate privacy-preserving design, enabling larger-scale studies.

* Clinical integration: Exploring the feasibility of integrating MoodJournal as a between-session monitoring tool in clinical or counselling settings, with appropriate ethical oversight.

* Personalised trajectory baselines: Adapting the trajectory engine to individual emotional baselines rather than absolute thresholds, improving sensitivity and specificity for users with atypical baseline emotional profiles.

## **5.3 Summary**

This Phase I submission has established the conceptual foundation, literature basis, research questions, proposed architecture, and planned methodology for MoodJournal. The project is well-grounded in peer-reviewed research, technically feasible within the proposed timeline, and addresses a genuine, well-documented gap in the NLP and digital mental health literature. Phase II will deliver the implementation, user study results, and empirical findings that test the project’s core research hypothesis.

# **References**

The following references follow APA 7th Edition format:

Brito, M. J., & Vieira-Andrade, J. D. (2025). Digital phenotyping in mental health — What can it mean for the future of psychiatry? European Psychiatry. https\://doi.org/10.1192/j.eurpsy.2025.477

Coppersmith, G., Dredze, M., & Harman, C. (2014). Quantifying mental health signals in Twitter. Proceedings of the ACL Workshop on Computational Linguistics and Clinical Psychology (CLPsych 2014), 51–60.

Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G., & Ravi, S. (2020). GoEmotions: A dataset of fine-grained emotions. Proceedings of the 58th Annual Meeting of the ACL, 4040–4054.

Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9(8), 1735–1780.

Jiang, Z., et al. (2024). Multimodal mental health digital biomarker analysis from remote interviews. IEEE Journal of Biomedical and Health Informatics, 28(3). https\://doi.org/10.1109/JBHI.2024

Kerz, E., Zanwar, S., Qiao, Y., & Wiechmann, D. (2023). Toward explainable AI (XAI) for mental health detection based on language behavior. Frontiers in Psychiatry, 14, 1219479\. https\://doi.org/10.3389/fpsyt.2023.1219479

Kim, Y., et al. (2024). Using large language models to detect depression from user-generated diary text data as a novel approach in digital mental health screening: Instrument validation study. Journal of Medical Internet Research, 26, e54617. https\://doi.org/10.2196/54617

Kroenke, K., Spitzer, R. L., & Williams, J. B. W. (2001). The PHQ-9: Validity of a brief depression severity measure. Journal of General Internal Medicine, 16(9), 606–613.

Li, A., Liu, R., Liu, X., & Han, J. (2023). Editorial: Improving the clinical value of digital phenotyping in mental health. Frontiers in Psychiatry, 14, 1251930\. https\://doi.org/10.3389/fpsyt.2023.1251930

Luo, L., Yang, H., & Chin, F. Y. L. (2018). EmotionX-DLC: Self-attentive BiLSTM for detecting sequential emotions in dialogue. arXiv:1806.07039.

Onnela, J. P., & Rauch, S. L. (2016). Harnessing smartphone-based digital phenotyping to enhance behavioral and mental health. Neuropsychopharmacology, 41(7), 1691–1696.

Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global vectors for word representation. Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), 1532–1543.

Pimpalkar, A., & Raj R, J. (2022). MBiLSTMGloVe: Embedding GloVe knowledge into the corpus using multi-layer BiLSTM deep learning model for social media sentiment analysis. Expert Systems with Applications, 203, 117581\. https\://doi.org/10.1016/j.eswa.2022.117581

Saravia, E., Liu, H.-C. T., Huang, Y.-H., Wu, J., & Chen, Y.-S. (2018). CARER: Contextualized affect representations for emotion recognition. Proceedings of EMNLP 2018, 3687–3697.

Suh, H., et al. (2023). A validation study of mental health monitoring through a mobile application using PHQ-9 and GAD-7. Psychiatry Investigation. https\://doi.org/10.30773/pi.2023.0047

Tan, M., Zhou, X., Shen, L., Li, Y., & Chen, X. (2024). Music’s dual role in emotion regulation: Network analysis of music use, emotion regulation self-efficacy, alexithymia, anxiety, and depression. Neural Plasticity, 2024, 1790168\. https\://doi.org/10.1155/2024/1790168

Zhang, T., Schoene, A. M., Ji, S., & Ananiadou, S. (2024). Natural language processing applied to mental illness detection: A narrative review. NPJ Digital Medicine. [https\://doi.org/10.1038/s41746-024](https://doi.org/10.1038/s41746-024)

**Dataset Info**

Yes. This is the dataset your app was built for, the Kaggle "Emotions dataset for NLP" mentioned in the About tab. It will train the model as-is. But it has some limits that matter for a journaling app and for your paper.

Why it fits

Same 6 labels. The labels are anger, fear, joy, love, sadness and surprise, the same as EMOTIONS in moodjournal\_app (1).py:71. Your list is in alphabetical order, which is also the order a scikit-learn LabelEncoder assigns. So the model's output positions will line up with the right emotion names.

Ready-made splits. You have train (\~16,000 rows), validation (\~2,000) and test (\~2,000).

Simple format. Each line is text;label, and the text is already lowercased with punctuation removed.

Limitations to know about

1.⁠ ⁠Uneven classes. In the training file, joy and sadness make up about 60% of the rows, while surprise is only about 3–4% and love about 8%. The model will lean towards predicting joy or sadness and will seldom predict surprise or love. A 90% overall accuracy can hide poor results on the small classes, so report per-class precision and recall (or macro-F1), and consider class weights during training.

2.⁠ ⁠Short sentences, not journal entries. Nearly every example is one short sentence starting with "i feel" or "im feeling", with one emotion. Journal entries are long and mix emotions ("work was stressful but lunch was nice"). That mismatch is exactly your research question 1, so the dataset works for training. To answer RQ1, though, you need some real journal-style text with its own labels as a separate test set.

3.⁠ ⁠Long entries get cut off at the start. The code pads with truncating='pre', so for an entry longer than maxlen words it keeps only the end. A simple fix is to split the entry into sentences, classify each one, and average the probabilities.

4.⁠ ⁠Your text cleaning removes negation. remove\_stop\_words uses NLTK's stopword list, which contains "not", "no", "nor", "don't" and similar words. So "i don't feel happy" becomes "feel happy". Some dataset labels already look wrong because of negation:

i didnt feel humiliated;sadness

im not feeling mellow;joy

Keep the negation words, e.g. self.stop\_words \-= {"not", "no", "nor"}, and retrain.

5.⁠ ⁠Some labels are inaccurate. For example, "i feel so cold" is labelled anger and "i was feeling a little vain" is labelled sadness. This is expected for this dataset and caps how accurate any model can get. It's worth one sentence in your limitations section.

6.⁠ ⁠No "neutral" option. Every entry is forced into one of the 6 emotions, so a plain entry like "went to class, had lunch" will still get a label, probably joy or sadness.

7.⁠ ⁠It can't support the PHQ-9 proxy. The dataset has no clinical scores. The PHQ-9 proxy and RQ2 need data from your planned user study, with participants' actual PHQ-9 and GAD-7 scores, not this dataset.

What's missing

There's no training script in this folder. The app only loads finished model files (bilstm\_emotion\_model.h5, tokenizer.pkl, and so on). I can write a training script for these three files: it would use the same text cleaning as the app (with the negation fix), GloVe 200d embeddings, class weights, a per-class evaluation report, and would save all the files the app expects.


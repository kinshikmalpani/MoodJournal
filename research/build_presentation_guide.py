"""Build the beginner-facing preparation guide from audited project results."""
from pathlib import Path
import json
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'deliverables'
OUT.mkdir(exist_ok=True)
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
sec.top_margin = sec.bottom_margin = Inches(.7)
sec.left_margin = sec.right_margin = Inches(.8)
sec.header_distance = sec.footer_distance = Inches(.3)
for name in ['Normal', 'Title', 'Subtitle', 'Heading 1', 'Heading 2', 'Heading 3']:
    style = doc.styles[name]
    style.font.name = 'Calibri'
    style.font.color.rgb = RGBColor(0, 0, 0)
    style.paragraph_format.space_after = Pt(7)
doc.styles['Normal'].font.size = Pt(11)
doc.styles['Normal'].paragraph_format.line_spacing = 1.08
doc.styles['Title'].font.size = Pt(28)
doc.styles['Heading 1'].font.size = Pt(20)
doc.styles['Heading 2'].font.size = Pt(13)
doc.styles['Heading 2'].paragraph_format.space_before = Pt(10)
header = sec.header.paragraphs[0]
header.text = 'MOODJOURNAL  |  PRESENTATION PREPARATION GUIDE'
header.runs[0].font.size = Pt(8)
footer = sec.footer.paragraphs[0]
footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
footer.add_run('Project snapshot 13 November 2026  •  ')
field = OxmlElement('w:fldSimple'); field.set(qn('w:instr'), 'PAGE'); footer._p.append(field)
for r in footer.runs: r.font.size = Pt(8)

def p(text, bold=False):
    paragraph = doc.add_paragraph()
    paragraph.add_run(text).bold = bold
    return paragraph

def h(text): doc.add_heading(text, level=2)
def page(title):
    doc.add_page_break()
    doc.add_heading(title, level=1)
def bullet(text): doc.add_paragraph(text, style='List Bullet')
def table(headers, rows, widths):
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for cell, text, width in zip(t.rows[0].cells, headers, widths):
        cell.text = text; cell.width = Inches(width)
    repeat = OxmlElement('w:tblHeader'); t.rows[0]._tr.get_or_add_trPr().append(repeat)
    for row in rows:
        cells = t.add_row().cells
        for cell, text, width in zip(cells, row, widths): cell.text = str(text); cell.width = Inches(width)
    borders = OxmlElement('w:tblBorders')
    for edge in ['top', 'left', 'bottom', 'right', 'insideH', 'insideV']:
        e = OxmlElement('w:' + edge); e.set(qn('w:val'), 'single'); e.set(qn('w:sz'), '4'); e.set(qn('w:color'), 'D9D9D9'); borders.append(e)
    t._tbl.tblPr.append(borders)
    for i, row in enumerate(t.rows):
        no_split = OxmlElement('w:cantSplit'); row._tr.get_or_add_trPr().append(no_split)
        for j, cell in enumerate(row.cells):
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            margins = OxmlElement('w:tcMar')
            for edge in ['top', 'left', 'bottom', 'right']:
                e = OxmlElement('w:' + edge); e.set(qn('w:w'), '90'); e.set(qn('w:type'), 'dxa'); margins.append(e)
            cell._tc.get_or_add_tcPr().append(margins)
            shade = OxmlElement('w:shd'); shade.set(qn('w:fill'), '233D4D' if i == 0 else ('F1F4F6' if i % 2 == 0 else 'FFFFFF')); cell._tc.get_or_add_tcPr().append(shade)
            for para in cell.paragraphs:
                para.paragraph_format.space_after = Pt(3)
                para.paragraph_format.line_spacing = 1.02
                if j > 0: para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for r in para.runs:
                    r.font.size = Pt(10)
                    if i == 0: r.bold = True; r.font.color.rgb = RGBColor(255,255,255)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)

doc.add_paragraph('MoodJournal presentation preparation guide', style='Title')
doc.add_paragraph('Understanding the application the research and the evidence', style='Subtitle')
p('Prepared for the project presenter • 13 November 2026')
p('This guide assumes you are starting with very little context. It explains what we built, why we built it, how the code works, what the experiments establish, and how to answer questions without overstating the results.')
h('The project in one paragraph')
p('MoodJournal is a local English journaling application that predicts emotional themes in text and organises those predictions over time. We compared a traditional text classifier with a compact pretrained transformer, built a working interface, and evaluated both on the original benchmark, independently sourced event descriptions, and fictional timelines. The transformer performs better than the baseline on the external text task, but neutral language, mixed emotions, and daily trends remain difficult. The project is an applied machine-learning study and reflection prototype, not a diagnostic tool.')
h('How to use this guide')
bullet('First pass: read sections 1–4, 8–10 and 13 to understand the idea, application flow, results and contribution.')
bullet('Second pass: read sections 5–7 and 11–12 to understand the experimental method and literature.')
bullet('Rehearsal: use sections 14–18 for the demo, slide sequence, spoken introduction and viva answers.')
bullet('Reference: sections 19–20 map the files and sources. Do not try to memorise every hyperparameter.')
h('The conclusion to remember')
p('A compact contextual model can improve emotion classification over our sparse baseline on the evaluated datasets. That advantage does not automatically establish reliable journal understanding or accurate tracking of a person’s mood.', True)
p('Read the performance tables with their dataset names. Our 91.50% accuracy belongs to the original short-text test; it is not a measured journal accuracy.')

page('1 The problem and the intended user')
h('What problem are we trying to address')
p('A person can write many journal entries and still find it difficult to review recurring emotional themes. Reading weeks of text manually takes effort. Our proposed solution adds a lightweight structure: it stores entries by date, predicts emotions in individual passages, and displays daily summaries. The intended user is an English-speaking person using a single computer for private reflection.')
p('This is a proposed use case, not a demonstrated wellbeing benefit. We have not run a usability study showing that people reflect more effectively, and we have not measured improvement in mental health. The engineering problem we have solved is building the local analysis and record-keeping workflow; the research problem is evaluating how well its classifiers behave.')
h('What the user actually does')
p('The user writes about their day, selects Analyse & save, reads the predicted themes and passage results, and later reviews their history. They can compare models, export entries, and delete individual entries. Inference runs on the computer, so the app does not need an external model service to analyse the text.')
h('Why someone might prefer this to a chatbot')
p('A general conversational AI is useful for open-ended discussion and may understand subtle context better. MoodJournal provides a ready-made routine: dated storage, automatic summaries, a calendar view and fixed model versions. The user does not have to repeatedly ask for the same analysis or assemble their own tracking workflow. These features are a product choice, not something other AI systems could never implement.')
p('We have not compared MoodJournal with ChatGPT or another conversational model on the same evaluation set. Do not claim that we are more accurate, more helpful, or categorically more private than every AI alternative. Local conversational models also exist.')
h('The boundary between text and experience')
p('A sentence about sadness can describe the writer, another person, a memory, a hypothetical situation, or an emotion that has stopped. Our classifier assigns a label from patterns in the words. It cannot establish the writer’s true internal state. This distinction explains why the app says model scores rather than emotional intensity.')

page('2 The concepts you need before the technical details')
h('A model learns a mapping from examples')
p('Supervised learning means giving a program examples with target labels. An example might be a sentence paired with joy. During training, the program adjusts numerical parameters so its predictions better match the labels. At inference time, it uses the learned parameters to predict a label for new text. Saving a journal entry does not retrain our model.')
h('Our six labels')
p('The supported categories are anger, fear, joy, love, sadness and surprise. They are the dataset’s categories; they do not cover all human emotion. In particular, the model has no trained neutral class. Do not describe this list as Ekman’s six emotions: that taxonomy includes disgust rather than love.')
h('Sentiment and emotion are different tasks')
p('Sentiment commonly separates positive, negative and neutral polarity. Emotion classification asks for more specific categories. Fear and anger can both be negative but are different predictions. Our model learns one of six classes for each training example. The app’s mixed label is added later by a rule; it is not a seventh class learned during training.')
h('Scores are relative model outputs')
p('For a neural classifier, softmax converts six raw output numbers into nonnegative scores that sum to one. A joy score of 0.80 means the model strongly favours joy among its available classes. It does not mean the person is 80% happy. A model can confidently select the wrong class, especially for neutral text or an emotion outside its taxonomy.')
h('Other terms used in the project')
table(['Term', 'Plain meaning'], [
('Token', 'A unit used by the model; it may be a word, a word piece or punctuation.'),
('Epoch', 'One pass through the training examples.'),
('Batch', 'A group of examples processed together during an update.'),
('Learning rate', 'Controls the size of parameter updates.'),
('Checkpoint', 'Saved model state; some checkpoints also preserve the optimizer and progress.'),
('Domain shift', 'New input differs from the data on which the model was developed.'),
('Ablation', 'A comparison that changes one component to study its effect.')], [1.3, 5.5])

page('3 How a journal entry moves through the system')
p('There are two distinct pipelines: training produces a saved model; the application loads that model to analyse new entries. The application does not train a transformer each time you click a button.')
h('The application pipeline')
p('Write text → validate input → split into passages → predict six scores per passage → create entry summary → save in SQLite → display history and daily summaries.', True)
p('Input validation rejects empty or nonverbal input and limits an entry to 12,000 characters. Passage splitting follows sentence or newline boundaries, then divides long passages into chunks of at most 80 words. The transformer tokenizer additionally uses windows of at most 128 tokens during inference so the full input is retained.')
p('Each passage receives scores and a top label. If its highest score is below the saved threshold, it can return uncertain. The baseline also checks whether a passage contains any recognised vocabulary. A high-confidence neutral sentence can still receive an emotion because the model has no neutral class.')
p('The entry’s score vector is the equal-weight mean of its passage vectors. If at least two distinct confident passage labels occur, the displayed summary is mixed. Otherwise, the summary uses the top averaged score subject to the threshold. Two sentences are given equal passage weight even if one is much longer; this is an experimental design choice.')
h('A worked example with illustrative numbers')
p('Suppose a journal says “I am worried about tomorrow. I am delighted that my project works.” If the first passage is confidently labelled fear and the second joy, the app displays a mixed summary. If their joy scores are 0.10 and 0.80, the entry joy score is (0.10 + 0.80) / 2 = 0.45. These numbers illustrate the arithmetic; they are not a recorded prediction.')
h('What gets saved')
p('SQLite stores the date, text, predicted labels, scores, passage results, model version and creation time. The date follows Asia/Kolkata. Switching models affects future analyses; existing saved predictions are not silently recomputed. Timeline comparisons are filtered to one model version so scores from different models are not merged.')
p('SQLite is a local database file. The current file is not encrypted, and this workspace is under OneDrive. Local inference means text is not sent to a model API; it does not guarantee that the surrounding filesystem is never synced.')

page('4 The two approaches we compared')
h('TF IDF with logistic regression')
p('TF-IDF turns text into a vector of word features. A feature gets more weight when it is useful for distinguishing a document rather than appearing everywhere. Our vectorizer uses single words and adjacent word pairs, such as “very happy”, with a vocabulary learned only from training text. Logistic regression then learns weights connecting these features to each emotion class.')
p('Despite its name, logistic regression is used here for classification. It is a useful baseline because it is inexpensive, fast, and provides a meaningful comparison against a more complex model. It can recognise informative phrases but does not model long contextual relationships in the way a transformer does. Word pairs help with some local context; they do not solve arbitrary negation.')
p('Implementation settings: up to 60,000 features, minimum document frequency 2, sublinear term frequency, unigrams and bigrams, regularisation setting C = 4, and up to 1,000 optimisation iterations. We compared ordinary and class-balanced training; the class-balanced candidate won on validation macro-F1.')
h('Partially fine tuned DistilBERT')
p('DistilBERT is a compact transformer pretrained on general English. Its token representations depend on surrounding words. Self-attention is the mechanism that lets tokens use information from other tokens. A classification head converts the resulting representation into scores for our six emotions. We downloaded pretrained weights and fine-tuned them; we did not pretrain a language model from scratch.')
p('We froze the embeddings and earlier encoder layers and trained the last two encoder layers plus the classifier. Freezing means those parameters are not updated. This reduces training work and keeps much of the pretrained representation. It does not guarantee that partial fine-tuning is better than full fine-tuning: we did not run that controlled comparison.')
h('Why we changed the original architecture plan')
p('The Phase I proposal described BiLSTM with GloVe. We ultimately implemented a sparse baseline and DistilBERT because a compact pretrained model offered a practical way to test contextual transfer with the available setup. We have no BiLSTM results and cannot claim DistilBERT beat it. DistilBERT’s original paper motivates its compact architecture, not our exact freezing policy [3].')

page('5 How the experiments developed')
h('First establish a trustworthy comparison')
p('We audited the original archive, removed duplicate and conflicting examples under a recorded policy, trained two logistic-regression candidates and selected one on validation data. We kept the unknown-source data.zip outside training. Personal journal entries and our AI-authored diagnostics were never used as training examples.')
h('The transformer checkpoint lineage')
table(['Stage', 'What happened', 'Saved model'], [
('Initial run', 'The run stopped during epoch two. Only the completed first epoch survived.', 'First epoch'),
('Continuation', 'Started from those weights with fresh optimizer state and trained three additional epochs.', 'Continued'),
('V2', 'Started from the best continued weights and trained three further epochs.', 'V2')], [1.0, 4.7, 1.1])
p('The final v2 weights therefore descend from 1 + 3 + 3 training epochs. This was not one uninterrupted seven-epoch run: optimizer and schedule settings changed between experiments. An optimizer maintains information about updates; restoring only weights is different from restoring that full training state.')
p('V2 used batch size 32, learning rate 0.00001, seed 42, a 96-token training limit, AdamW, and a warmup/decay learning-rate schedule. Its third epoch had the highest validation macro-F1. A seed controls random choices; one seed does not establish robustness across repeated training runs.')
h('Why validation comes before test')
p('The training set updates model parameters. Validation helps choose configurations, checkpoints and thresholds. The test set measures the chosen candidate. We selected models using validation macro-F1, not the test score. Because several checkpoints have been evaluated on the same original test set, those results are repeated benchmark measurements rather than an untouched external assessment.')
h('Resuming and checkpoint details')
p('The continuation script saves weights, optimizer and scheduler state, random-number states and batch progress. A test verifies an identical next update after restoration on CPU. The separate v2 script saves its best inference weights but does not save full resumable optimizer state. Do not say every training script supports exact resume.')

page('6 Datasets and the meaning of independence')
table(['Resource', 'Purpose', 'What it cannot establish'], [
('Original Emotions Dataset.zip', 'Training and six-class benchmark evaluation', 'Real-journal or clinical validity'),
('41 AI-authored development cases', 'Inspect neutral, mixed, long and contextual cases', 'Independent population accuracy'),
('enISEAR external descriptions', 'Evaluate transfer using existing human-generated data and labels', 'Longitudinal journal accuracy'),
('32 synthetic timeline entries', 'Controlled analysis of daily aggregation and failures', 'Real-user emotional trajectories'),
('Personal SQLite journal', 'Use the app', 'It is not our training or evaluation dataset')], [1.8, 2.7, 2.3])
h('Original data preparation')
p('The archive originally contained 16,000 training, 2,000 validation and 2,000 test examples. Text was lowercased and whitespace normalised for duplicate detection. Conflicting same-text labels were excluded within a split. Test had precedence over validation, which had precedence over training, so overlap was removed from earlier learning stages. The retained sizes are 15,923 / 1,993 / 2,000. This is our cleaned protocol, not an exact reproduction of a published split.')
p('The archive’s checksum records which bytes we used. A checksum is a digital fingerprint; it is not proof of the original download source or licence. The original archive’s provenance remains incompletely verified. Do not attribute it to a specific paper merely because its categories look familiar.')
h('External does not mean completely unknown to every model')
p('enISEAR was collected by other researchers and not used for our task-specific training. We checked for exact normalised overlap with all original benchmark splits and found none. That does not rule out near-duplicates or exposure during the pretrained language model’s development. “Externally sourced evaluation” is more precise than “guaranteed unseen text”.')
p('Synthetic labels are the intended meanings of fictional examples written during this project. They let us test designed cases, but are not independent human judgments. Assigning dates to unrelated external descriptions would not turn them into genuine longitudinal records.')

page('7 Understanding the evaluation metrics')
h('Accuracy')
p('Accuracy is the proportion of examples given the correct class. V2’s 91.50% accuracy on 2,000 original test examples means 1,830 were correctly classified. Accuracy can hide poor performance on a rare class if common classes dominate the dataset.')
h('Precision recall and F1')
p('For one emotion, precision asks: of the examples predicted as this emotion, how many were correct? Recall asks: of all examples with this true emotion, how many did the model find? F1 combines precision and recall using their harmonic mean, which penalises a poor value on either side.')
p('Example: if the model predicts fear 10 times and 8 are correct, precision is 8/10 = 0.80. If 16 examples truly express fear, recall is 8/16 = 0.50. F1 is 2 × 0.80 × 0.50 / (0.80 + 0.50), approximately 0.615. This is an illustrative example.')
h('Macro F1')
p('Macro-F1 computes each class’s F1 and averages them equally. It gives a rare class the same weight as a common one, which is useful for our imbalanced dataset. Our original benchmark averages six classes. The external primary score averages only the four overlapping classes. Their percentages are not interchangeable measures of the same task.')
h('Confusion matrices and uncertainty')
p('A confusion matrix counts true labels against predicted labels. It exposes patterns such as love being mistaken for joy. A score threshold can cause the app to abstain by showing uncertain. Choosing a threshold on validation data is not probability calibration and does not train the model to recognise neutral writing.')
h('What the external interval means')
p('We repeatedly resampled the enISEAR authors as groups, retaining related entries together, and recalculated v2’s macro-F1 difference from the baseline. The middle 95% of 1,000 bootstrap differences gives the reported interval. This quantifies sampling uncertainty conditional on this corpus and method. It does not account for every source of bias or prove that all future users will see the same improvement.')
p('Percentage points describe differences between percentages: 54.32% minus 36.23% is 18.09 percentage points. Do not call that an 18.09% relative increase.')

page('8 Results on the original benchmark')
table(['Model', 'Validation macro F1', 'Test accuracy', 'Test macro F1'], [
('TF-IDF baseline', '86.10%', '88.25%', '83.97%'),
('DistilBERT first epoch', '82.41%', '84.35%', '78.54%'),
('Continued DistilBERT', '88.83%', '90.30%', '86.36%'),
('DistilBERT v2', '89.01%', '91.50%', '87.61%')], [2.4,1.5,1.4,1.5])
h('What this shows')
p('V2 is the strongest checkpoint on the original validation criterion and test measurements. Compared with the baseline, it improves test accuracy by 3.25 percentage points and six-class macro-F1 by about 3.64 points. Compared with the continued model, it classifies 24 more of the 2,000 test examples correctly.')
p('The validation improvement from continued to v2 is only about 0.18 points. We did not repeat training across multiple seeds, so we should not interpret this small difference as a robust universal gain. The first recovered epoch is worse than the baseline; the result depends on training stage, not merely the model name.')
h('Why v2 is the current default')
p('The app chooses the completed model with the highest original validation macro-F1. That rule currently selects v2. Later external findings did not change this choice. Keeping that rule prevents us from silently using the external test set as a model-selection set. A future product revision can reconsider the choice, but it needs a new validation plan.')
h('Earlier journal development diagnostics')
p('The 41 AI-authored development cases included 28 single-emotion entries, six neutral cases, six mixed cases and one artificial length case. On the 28 single-emotion cases, v2 scored 81.10% passage macro-F1 and 85.56% whole-entry macro-F1. The continued model scored 78.28% and 96.63%, respectively. Both failed to abstain on all six neutral cases.')
p('These examples initially exposed that further short-text training did not improve every kind of input. They were useful debugging evidence, but too small and too dependent on our own example construction to establish real-journal performance. That limitation motivated the separate external evaluation.')

page('9 Results on external human generated text')
h('What enISEAR contains')
p('Troiano, Padó and Klinger collected prompted descriptions of emotional events through crowdsourcing [10]. Our targets are the original author prompt labels; reader votes are retained for inspection. These are human-generated descriptions and existing labels, not annotations produced by our model. They are reflective event descriptions, not dated private journals.')
p('The archive contains 1,001 English examples. We removed two identical-text duplicates with matching labels, retaining 999. The primary task uses 570 examples: anger 143, fear 143, joy 142 and sadness 142. The 429 disgust, guilt and shame examples remain in a separate unsupported-label analysis. We did not relabel them as sadness. The model could still predict any of its six classes; outputs were not restricted to the four targets.')
table(['Model', 'Passage macro F1', 'Whole entry macro F1'], [
('TF-IDF baseline', '36.23%', '41.29%'),
('Continued DistilBERT', '56.25%', '56.84%'),
('DistilBERT v2', '54.32%', '59.29%')], [2.8,2.0,2.0])
h('The defensible conclusion')
p('Both transformer checkpoints outperform the baseline on this external four-class task. V2 improves the primary passage score by 18.09 points; the paired author-group bootstrap interval is +12.57 to +23.16 points using 63 author groups. This is evidence of transfer to this particular external corpus, not proof of accuracy on all journal writing.')
p('Continued DistilBERT is best for passage averaging, while v2 is best for whole-entry inference. Preserving context may matter, but that explanation is an interpretation to investigate rather than an experimentally isolated cause. We did not tune aggregation or retrain after seeing these scores.')
h('Unsupported emotions reveal another weakness')
p('On 429 unsupported-label examples, the baseline abstained 160 times, the continued model 11 times, and v2 only 5 times. These cases are not neutral; they express emotions our classifier cannot name. Confident assignment to an available class can therefore be misleading. The high original benchmark score does not remove this limitation.')

page('10 Daily summaries and the synthetic timelines')
h('How the app chart works')
p('Each entry has a mean passage-score vector. For a day with multiple entries, the app averages those entry vectors equally. Dates with no entries remain unobserved. The chart uses only one model version at a time. This produces a reproducible summary of predictions, but averaging predictions does not make them accurate measurements of mood.')
p('Illustration: if two entries have sadness scores 0.20 and 0.80, the daily sadness score is 0.50. The same average could arise from a consistently moderate score or two very different entries. Daily averages can conceal within-day changes, and the number of entries matters when interpreting the chart.')
h('What the new stress test measures')
p('We constructed 32 fictional entries across four people and 28 observed person-days. Two timelines have a missing calendar day; each person has one day with multiple entries. The examples include worry, recovery, conflict, affection, surprise, neutral activity descriptions and negation. They were frozen before inference; their intended passage labels are AI-authored, not independent human ground truth.')
p('For this experiment we computed passage-label frequencies rather than average softmax scores: the fraction of passages assigned to each emotion, averaged equally by entry and then by day. This is a separate diagnostic and does not directly validate the app’s score chart. We compare adjacent calendar days only, using a fixed 0.05 change deadband.')
table(['Synthetic measure', 'Baseline', 'Continued', 'V2'], [
('Direction correct on 40 changed class transitions', '27.5%', '62.5%', '60.0%'),
('Direction correct on 92 unchanged class transitions', '84.8%', '82.6%', '76.1%'),
('Daily frequency mean absolute error', '0.1220', '0.1101', '0.1250'),
('Neutral entries abstained', '2/6', '1/6', '0/6'),
('Mixed emotion sets exactly recovered', '0/8', '2/8', '2/8')], [3.2,1.2,1.2,1.2])
p('Lower daily error is better. V2 recovers more designed changes than baseline, but makes more false changes and has slightly higher daily error. We cannot claim reliable trend tracking. The six class transitions within a day pair are related observations, not independent participants.')

page('11 What we researched and what literature contributed')
h('Our research question')
p('We investigated whether a compact contextual classifier improves on a sparse baseline for six-class emotion recognition, and how its behaviour changes when applied to reflective text and synthetic sequences. Research here means a documented question, a controlled comparison, measured outcomes, and analysis of what those outcomes do and do not support.')
h('The literature review method')
p('We conducted a focused narrative review using primary sources from ACL Anthology, arXiv, PMLR and JMIR. We checked source records and selected methods, results and limitations. The core review contains nine works, with enISEAR added as the external dataset source. This is not an exhaustive systematic review or complete coverage of 2025–2026 research. Some literature was reviewed after implementation to assess and improve our rationale; do not claim every paper originally caused a design decision.')
h('Sparse baselines and contextual representations')
p('Wang and Manning studied strong simple classifiers and bigram features for sentiment and topic tasks [1]. Their work supports the practice of using a meaningful inexpensive baseline. Our TF-IDF/logistic-regression implementation is not their NB-SVM algorithm, and sentiment results do not directly validate emotion classification.')
p('CARER develops contextual affect representations using graph-derived patterns enriched with embeddings and a classifier [2]. It supplies related work on representations for emotion. We did not implement CARER or reproduce its experiments. Its name in a citation also does not verify where our original ZIP came from.')
p('DistilBERT studies a smaller pretrained transformer created through knowledge distillation [3]. We use an existing pretrained model and fine-tune it. We did not perform the original distillation procedure. Its language benchmarks support an efficiency rationale, but our own experiments are needed to assess emotion prediction.')
h('Annotation choices matter')
p('GoEmotions provides human-labelled Reddit comments with a broader emotion taxonomy and neutral [4]. Its relevance is the need to define labels and account for multiple emotions and neutral text. We did not train on GoEmotions. Its target-domain fine-tuning experiments should not be presented as a promise that an arbitrary pretrained model works on journals without validation.')

page('12 Literature that shaped the evaluation')
h('Domain transfer')
p('Zanwar and colleagues combine transformer models with psycholinguistic features and evaluate transfer across emotion datasets [5]. Their work directly motivates separating source-benchmark results from external performance. Their hybrid model is not our architecture; we use the evaluation lesson rather than claiming to reproduce their method.')
p('Koufakou and colleagues study a unified emotion corpus and a distilled RoBERTa model with repeated experiments [9]. Differences in corpora, label mappings and protocols make cross-paper percentages hard to compare. For our external test, we therefore state the four overlapping labels and retain unsupported labels separately. Repeated training seeds remain future work for us.')
h('Confidence and behavioural failures')
p('Guo and colleagues investigate neural-network calibration and temperature scaling [6]. This supports separating prediction confidence from measured correctness. We did not implement temperature scaling. Our threshold is a validation-selected abstention rule, not a calibrated probability estimate or a neutral detector.')
p('CheckList proposes minimum-functionality, invariance and directional tests for NLP systems [8]. Our existing diagnostic examples are consistent with the motivation to inspect specific failures, but we have not implemented the full CheckList methodology. A useful next step is to test controlled changes in negation, names, quoted speech and paraphrases, with expectations fixed before running the models.')
h('Why diary studies do not validate our clinical claims')
p('Shin and colleagues studied LLM analysis of diaries with questionnaire-based assessments from participants [7]. Their study has a different target and a participant evidence base that our project does not have. It motivates independent evaluation, not the claim that our six emotion scores diagnose depression. Its published correction concerns a funding grant number rather than model results.')
h('How to explain the research gap honestly')
p('Do not say nobody has analysed journals before. The narrower question is whether our compact local classifier and aggregation method work adequately in this setting. Our contribution is evidence from our controlled comparison and implementation. Literature establishes the surrounding methods and known difficulties; our experiments establish only our measured results.')
p('In your report, distinguish “implemented”, “motivated by related work” and “proposed future work”. We did not implement a novel neural architecture, validated clinical scale, or clinical intervention.')

page('13 Contribution limitations and the next improvements')
h('What is our contribution')
p('We built a functioning local workflow and compared a sparse classifier with partially fine-tuned DistilBERT under shared duplicate controls. We preserved model-versioned predictions, tested calendar aggregation, and added externally sourced evaluation and synthetic sequence diagnostics. The finding that checkpoint and aggregation choices behave differently outside the original benchmark is a useful empirical result.')
p('These contributions are appropriate for an applied minor project. None establishes that the app is globally unique. Its practical distinction is a dedicated, reproducible local journal workflow. We have not conducted a comprehensive competitor analysis or tested superiority to a general-purpose AI assistant.')
h('What remains unproven')
bullet('Accurate classification of naturally written, independent journals, especially neutral and unsupported emotions.')
bullet('Reliable agreement between daily textual predictions and a person’s actual emotional experience.')
bullet('Clinical benefit, depression detection, early warning, or replacement of professional assessment.')
bullet('Robustness across training seeds, languages, demographics and unseen writing styles.')
p('The optional three-day negative-score reflection prompt is an unvalidated rule: daily anger + fear + sadness score sum at least 0.6 on three consecutive days ending today. It is not a clinically derived threshold. Do not use it as the central research claim.')
h('Priorities for a stronger next version')
p('First, collect consented independent journals with two annotators and keep a final test set separate by author. This provides evidence that more model training alone cannot supply. Next, compare whole-entry, passage and context-preserving aggregation on development data, then test the selected design on fresh data. Our external results already suggest this deserves investigation.')
p('Then address unsupported/neutral inputs and calibration using appropriate labelled development data; merely increasing a threshold is not a complete solution. Add optional user self-reports as a separate field to study disagreement, never as a silent replacement for predictions. Finally, repeat seeds and evaluate usability, storage protection and broader language support. These are proposals, not completed features.')

page('14 Your demonstration step by step')
h('Start the fictional demo')
p('Open PowerShell in the MoodJournal project folder and run the following command. It uses the existing environment and model artifacts; no retraining is needed.')
p('powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\\run.ps1 -Demo', True)
p('Open http://127.0.0.1:8502. Keep the terminal open; Ctrl+C stops the server. Demo mode uses local_data/demo.sqlite3 and preserves existing demo entries. Its seven seeded fictional entries are separate from the 32-entry synthetic evaluation set and from the personal journal. If the port is already in use, use the existing demo or stop the previous server from its own terminal.')
h('A five minute demonstration')
bullet('Show the demo banner and explain that the visible entries are fictional. Confirm the selected model is v2.')
bullet('Enter a short example of your own. Click Analyse & save. Explain the leading label and that the bars are relative model scores.')
bullet('Use two sentences with different expressed emotions. Expand the individual passages. Explain that mixed is a rule based on differing confident passage predictions, not a trained seventh class.')
bullet('Open Timeline & history. Point out dates, missing observations, multiple entries and the model-version selector. Do not claim the chart measures emotional intensity.')
bullet('Show model comparison in About the project. The app shows the original benchmark and earlier development comparison; the newer external and timeline results are in the saved research reports and this guide.')
bullet('Switch to the baseline to demonstrate the comparison workflow. Existing saved entries retain their original model version. Delete a newly added fictional entry or show export.')
h('Show one limitation deliberately')
p('Use “I do not resent my brother. I adore him and treasure our closeness.” Explain that negation and splitting may cause a misleading anger signal. Do not promise a particular output from memory; inspect what the selected model actually produces. A limitation demonstrated honestly is more useful than hiding a surprising prediction.')
h('Before presenting')
p('Launch and practise once beforehand, keep the trained artifacts available, and use the demo database. Do not display personal entries or share the older Mac transfer ZIP, which contains personal data. Keep this guide and the result JSON files available if asked about the evidence.')

page('15 A twelve slide presentation storyline')
p('For a ten-to-twelve-minute talk, spend most of the time on the question, method and results. The demo supports the argument; it should not consume the whole presentation.')
table(['Slide', 'Main message and what to show'], [
('1  Title', 'MoodJournal — evaluating a compact emotion classifier for local journal reflection. State the scope in one sentence.'),
('2  Problem', 'Show a fictional week of writing. Explain the proposed need to review emotional themes.'),
('3  Dedicated workflow', 'Explain dated storage, local inference, passage inspection and export. Acknowledge conversational AI as an alternative.'),
('4  Literature', 'Group papers by baselines, contextual models, transfer and validation. Show how these motivate our questions.'),
('5  Research questions', 'Baseline versus transformer; transfer to reflective text; behaviour of daily aggregation.'),
('6  Data and protocol', 'Shared cleaned splits, validation selection, separate external and synthetic evaluations.'),
('7  Architecture', 'Trace text through passages, model, summary, SQLite and chart. Separate training from inference.'),
('8  Original results', 'Show the six-class benchmark table. V2 reaches 91.50% accuracy and 87.61% macro-F1.'),
('9  External results', 'Show the four-class enISEAR table. Both transformers beat baseline; the winning checkpoint depends on aggregation.'),
('10  Demo and trends', 'Show the app and synthetic results. Explain changed versus unchanged transitions and neutral failures.'),
('11  Limitations', 'No independent real-journal or clinical validation; unsupported emotions and context remain difficult.'),
('12  Contribution', 'Summarise implemented workflow, empirical comparison and the next independent evaluation. Put full references in an appendix.')], [1.5,5.3])
p('If time is short, shorten the interface tour rather than omit the external results or limitations. Those results distinguish the presentation from a software-only demonstration.')

page('16 A spoken introduction and result explanation')
h('Opening script')
p('“Our project is MoodJournal, a local journaling application supported by an emotion-classification study. A journal can contain useful recurring themes, but reviewing weeks of writing manually takes effort. We explored whether a compact model could add a structured view of expressed emotions while keeping inference on the user’s computer.”')
p('“We compared TF-IDF with logistic regression against partially fine-tuned DistilBERT. We used the same cleaned dataset and selected checkpoints using validation macro-F1. The app analyses passages, saves versioned predictions, and displays daily summaries. Our goal is reflection and analysis of text; we are not diagnosing mental health conditions.”')
h('Transition into the research')
p('“A high short-text score does not necessarily mean the model understands journals. The literature on domain transfer motivated us to evaluate that separately. We used an externally collected human-labelled event-description corpus, and a separate synthetic set with fictional dated entries to test temporal behaviour.”')
h('Explain the results without overselling')
p('“V2 achieved 91.50% accuracy and 87.61% macro-F1 on our original six-class test. On the external four-class task, its passage macro-F1 was 54.32%, compared with 36.23% for the baseline. The previous continued checkpoint was slightly better with passage averaging, while v2 was best with whole-entry inference.”')
p('“This tells us that the transformer advantage transfers to that external corpus, but further training and splitting choices do not improve every setting. In synthetic timelines, v2 detected more intended changes than the baseline but also made more false changes and failed all six neutral examples. Therefore, we report a useful prototype and comparative evidence, not reliable tracking of a person’s actual mood.”')
h('Closing script')
p('“Our contribution is a reproducible comparison integrated into a working local application, together with explicit transfer and failure analysis. The next priority is independent journal annotation and testing of aggregation methods on fresh data. That is the evidence required before making a stronger claim about real journal use.”')
p('Use these paragraphs as a rehearsal aid, not as a speech to read mechanically. Be able to explain what each dataset and percentage represents in your own words.')

page('17 Likely viva questions about the research')
for question, answer in [
('What is new about your project', 'The contribution is an applied comparison and integrated local journal workflow with recorded provenance, model versions and failure analysis. We do not claim a new neural architecture or global product novelty.'),
('Why should someone not just talk to AI', 'They can. Our app provides a dedicated routine for dated entries, consistent local inference and automatic summaries. We have not shown it is more accurate than a conversational AI.'),
('Why DistilBERT rather than BiLSTM', 'We chose an available compact pretrained contextual model to test transfer under our constraints. BiLSTM was in the proposal but was not implemented, so we cannot claim an experimental victory over it.'),
('What exactly did you train', 'We trained the TF-IDF/logistic-regression baseline and fine-tuned the last two encoder layers plus classifier of pretrained DistilBERT. We did not pretrain or distil a language model ourselves.'),
('Did all approaches use the same data', 'The trained baseline and transformer variants used the same cleaned original splits. External enISEAR and synthetic cases were evaluation-only. data.zip and personal journals were excluded.'),
('What proves the model works on journals', 'We have limited transfer evidence on externally sourced reflective event descriptions, plus diagnostic examples. We do not have an independent real-journal participant study.'),
('What proves your daily trends are accurate', 'Tests verify computation and synthetic cases expose prediction behaviour. V2 has mixed trend results. We have not established agreement with real users’ daily emotional states.'),
('Why show negative results', 'They identify where deployment assumptions fail. A research comparison is useful even when extra training or a chosen aggregation rule does not improve every endpoint.')]:
    h(question); p(answer)

page('18 Likely viva questions about implementation and numbers')
for question, answer in [
('Why is macro F1 useful', 'It gives each emotion equal weight rather than letting common classes dominate the summary. Accuracy is still reported, but it is not the only criterion.'),
('Is 91 point 5 percent your journal accuracy', 'No. It is accuracy on 2,000 original short-text test examples. External event-text results and synthetic results are reported separately with different targets.'),
('Why did you evaluate only four external classes', 'enISEAR shares anger, fear, joy and sadness with our model. We retained disgust, guilt and shame as unsupported cases rather than invent a mapping. Love and surprise have no targets in that subset.'),
('Why does v2 remain the default if continued is better externally', 'The existing choice uses original validation macro-F1. Switching based on the external test would make it part of development. A future revision should state that decision and obtain a fresh evaluation.'),
('Is mixed a trained label', 'No. Each passage is scored over six classes. If confident passages have different labels, the app displays mixed. This heuristic can miss or overstate mixed emotions.'),
('Does a score measure how strongly the user feels', 'No. It is a relative classifier output. Neither probability calibration nor emotion-intensity validation has been established.'),
('Can the system diagnose depression or compute PHQ 9', 'No. The final implementation does not compute a clinical score. The original proposal’s proxy was not validated and is not a completed feature.'),
('What would you improve first', 'Independent journal labels and evaluation, then context-preserving aggregation and unsupported/neutral handling. More epochs alone do not address those evidence gaps.')]:
    h(question); p(answer)

page('19 Where the implementation and evidence live')
table(['File or folder', 'What to look for'], [
('app.py', 'Streamlit interface, saving flow, timeline, model selector and About tab.'),
('journal_core.py', 'Passage splitting, aggregation, SQLite storage and calendar logic.'),
('model_runtime.py', 'Local model loading and transformer token-window inference.'),
('model_catalog.py', 'Available models and validation-based default selection.'),
('train_baseline.py', 'Original data cleaning and logistic-regression comparison.'),
('train_transformer.py', 'Original transformer training experiment.'),
('continue_transformer.py and training_state.py', 'Continuation training and resumable state.'),
('train_transformer_v2.py', 'Three further epochs and best-validation selection.'),
('evaluate_models.py', 'Earlier 41-case AI-authored development comparison.'),
('research/prepare_pilot.py', 'External data preparation and synthetic timelines.'),
('research/evaluate_pilot.py', 'Frozen external and synthetic evaluation metrics.'),
('research/pilot/protocol.json', 'Counts, exclusions, frozen endpoints and model/data hashes.'),
('artifacts/pilot_evaluation/', 'Summary and individual predictions for all three evaluated models.'),
('README.md and research/references.bib', 'Current methodology, results, caveats and bibliographic records.')], [2.8,4.0])
h('How to verify the project')
p('The final app integration passed 10 tests. Three additional tests for the pilot metrics then passed separately; do not claim a newly rerun combined suite from that fact alone. Run the full suite with .\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v. Tests cover saving/deletion, preservation across model changes, calendar logic, split overlap, checkpoint restoration, demo isolation and pilot calculations.')
p('The Phase I synopsis is historical. It mentions BiLSTM/GloVe, a clinical proxy, Spotify, word attribution and a participant study that were not completed. Use the current code and result artifacts as the authority for what was actually implemented.')

page('20 References and your final preparation checklist')
references = [
('1', 'Wang and Manning (2012). Baselines and Bigrams: Simple, Good Sentiment and Topic Classification.', 'https://aclanthology.org/P12-2018/'),
('2', 'Saravia et al. (2018). CARER: Contextualized Affect Representations for Emotion Recognition.', 'https://aclanthology.org/D18-1404/'),
('3', 'Sanh et al. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.', 'https://arxiv.org/abs/1910.01108'),
('4', 'Demszky et al. (2020). GoEmotions: A Dataset of Fine-Grained Emotions.', 'https://aclanthology.org/2020.acl-main.372/'),
('5', 'Zanwar et al. (2022). Improving the Generalizability of Text-Based Emotion Detection by Leveraging Transformers with Psycholinguistic Features.', 'https://aclanthology.org/2022.nlpcss-1.1/'),
('6', 'Guo et al. (2017). On Calibration of Modern Neural Networks.', 'https://proceedings.mlr.press/v70/guo17a.html'),
('7', 'Shin et al. (2024). Using Large Language Models to Detect Depression From User-Generated Diary Text Data as a Novel Approach in Digital Mental Health Screening: Instrument Validation Study.', 'https://www.jmir.org/2024/1/e54617/'),
('8', 'Ribeiro et al. (2020). Beyond Accuracy: Behavioral Testing of NLP Models with CheckList.', 'https://aclanthology.org/2020.acl-main.442/'),
('9', 'Koufakou et al. (2024). Towards a new Benchmark for Emotion Detection in NLP: A Unifying Framework of Recent Corpora.', 'https://aclanthology.org/2024.genbench-1.13/'),
('10', 'Troiano, Padó and Klinger (2019). Crowdsourcing and Validating Event-focused Emotion Corpora for German and English.', 'https://aclanthology.org/P19-1391/'),
]
for number, title, url in references:
    para = p(f'[{number}] {title}\n{url}')
    para.paragraph_format.space_after = Pt(6)
    for r in para.runs: r.font.size = Pt(9)
h('Before you walk into the presentation')
p('Be able to draw the entry flow, distinguish training from inference, name the three evaluation settings, explain why macro-F1 matters, and identify one success and one failure. Practise the demo in the fictional database. If asked something we did not test, say so and explain the experiment that would answer it.')
p('Your strongest position is specific: we built and evaluated a local prototype, observed a transformer advantage on external event text, and found limitations that define the next research step.')

doc.core_properties.title = 'MoodJournal presentation preparation guide'
doc.core_properties.subject = 'Beginner guide to implementation research evaluation and presentation'
doc.core_properties.author = 'MoodJournal project'
doc.save(OUT / 'MoodJournal_Presentation_Preparation_Guide.docx')
print(OUT / 'MoodJournal_Presentation_Preparation_Guide.docx')

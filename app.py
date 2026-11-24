"""Run with: python -m streamlit run app.py"""
import json
import pandas as pd
import streamlit as st
from journal_core import ROOT, today, analyse, save_entry, load_entries, delete_entry, daily_summary, negative_streak
from model_runtime import load_artifact
from model_catalog import MODELS, available_models, recommended_model, read_report
from journal_core import DEMO_MODE

st.set_page_config(page_title='MoodJournal', page_icon='📓', layout='centered')
st.title('📓 MoodJournal')
st.caption('A space to write, reflect, and explore emotions in your words.')
if DEMO_MODE:
    st.info('Demo journal: fictional examples, stored separately from your personal journal.')


@st.cache_resource
def load_model(name):
    return load_artifact(name)


comparison_path = ROOT / 'artifacts' / 'model_comparison.json'
comparison = json.loads(comparison_path.read_text(encoding='utf-8')) if comparison_path.exists() else None
available = available_models()
if not available:
    st.error('No completed model found. Follow the model setup instructions in README.md.')
    st.stop()
recommended = recommended_model(available)
with st.sidebar:
    st.subheader('Model comparison')
    model_name = st.selectbox('Analysis model', available, index=available.index(recommended), format_func=lambda name: MODELS[name][0])
    st.caption('Research controls: changing the model affects new entries only. Saved predictions remain unchanged.')
if st.session_state.get('active_model') != model_name:
    st.session_state.pop('last_result', None)
    st.session_state['active_model'] = model_name


try:
    artifact = load_model(model_name)
except FileNotFoundError:
    st.error('The model has not been trained. Run python train_baseline.py first.')
    st.stop()

journal, timeline, about = st.tabs(['Write', 'Timeline & history', 'About the project'])
with journal:
    st.caption(today().strftime('%A, %d %B %Y'))
    with st.form('journal_form'):
        text = st.text_area('How was your day?', height=180, max_chars=12000, placeholder='There were a few difficult moments today, but…')
        submitted = st.form_submit_button('Analyse & save', type='primary')
    if submitted:
        try:
            result = analyse(text, artifact)
            save_entry(text, result)
            st.session_state['last_result'] = result
            st.success('Entry saved on this computer.')
        except ValueError as error:
            st.warning(str(error))
    if 'last_result' in st.session_state:
        result = st.session_state['last_result']
        st.subheader('What your words suggest')
        if result['label'] == 'mixed':
            st.write('Different passages suggest **' + ', '.join(result['detected_emotions']) + '**.')
        else:
            st.write('No clear single emotion detected.' if result['label'] == 'uncertain' else f"Leading emotion: **{result['label'].capitalize()}**")
        st.bar_chart(pd.DataFrame({'Model score': result['scores']}))
        st.caption('Scores describe the model’s predictions, not the intensity of your feelings. Uncertain does not mean neutral.')
        with st.expander('Explore individual passages', expanded=True):
            for part in result['passages']:
                st.text(part['text'])
                st.caption(f"{part['label'].capitalize()} · top model score {part['score']:.0%}")

with timeline:
    days = st.selectbox('Time window', [7, 14, 30], index=1)
    entries = load_entries(days=days)
    versions = sorted({entry['result']['version'] for entry in entries})
    if versions:
        chart_version = st.selectbox('Timeline model', versions, index=versions.index(artifact['version']) if artifact['version'] in versions else 0)
        daily = daily_summary([entry for entry in entries if entry['result']['version'] == chart_version])
        st.caption('The chart uses one model version at a time so scores from different models are not mixed. History below includes every entry.')
    else:
        daily = []
    if daily:
        frame = pd.DataFrame(daily).set_index('day')
        dates = pd.date_range(end=today(), periods=days).strftime('%Y-%m-%d')
        chart = frame.reindex(dates)
        st.line_chart(chart[list(artifact['pipeline'].classes_)])
        st.caption('Daily averages give each entry equal weight. Missing days have no observations. Scores are not clinical measurements.')
        st.metric('Days with entries', len(daily))
        streak = negative_streak(daily)
        if streak >= 3:
            st.info(f'Your writing has contained repeated negative-emotion signals across {streak} consecutive days. You might take a moment to reflect on what support would feel helpful. This is an experimental reflection prompt.')
        for entry in reversed(entries):
            with st.expander(f"{entry['day']} · {entry['result']['label'].capitalize()} · Entry {entry['id']}"):
                st.text(entry['text'])
                st.caption(f"Model: {entry['result']['version']}")
                if st.button('Delete this entry', key=f"delete_{entry['id']}"):
                    delete_entry(entry['id'])
                    st.session_state.pop('last_result', None)
                    st.rerun()
    else:
        st.info('Your timeline will appear here after you save an entry.')
    all_entries = load_entries()
    if all_entries:
        st.download_button('Export all entries (JSON)', json.dumps(all_entries, indent=2, ensure_ascii=False), file_name='moodjournal-export.json', mime='application/json')

with about:
    continued_progress = ROOT / 'artifacts' / 'distilbert_continued' / 'progress.json'
    if continued_progress.exists():
        try:
            status = json.loads(continued_progress.read_text(encoding='utf-8'))
            with st.expander('Continuation training status', expanded=status['phase'] != 'complete'):
                st.write(f"Latest recorded phase: {status['phase']}. Completed training steps: {status['global_step']}. Best validation macro-F1: {status['best_validation_macro_f1']:.3f}.")
                st.caption('Saved snapshot, not a live process-health check. Refresh this page for the latest snapshot.')
        except (OSError, json.JSONDecodeError):
            pass
    st.write('MoodJournal is a student research prototype for English emotion classification and journal reflection. It does not diagnose mental health conditions.')
    st.write('Compare a TF-IDF/logistic-regression baseline with a locally trained DistilBERT checkpoint. DistilBERT uses partial fine-tuning of its final two encoder layers and classifier for this CPU experiment.')
    st.write('The six trained categories are anger, fear, joy, love, sadness, and surprise. Neutral, sarcasm, unfamiliar language, and mixed emotions remain limitations. Journal-level aggregation has not yet been independently validated.')
    st.write('Entries are stored in a local SQLite file without encryption. Inference runs locally; journal text is not sent to a model service. This app is designed for one user on this computer.')
    progress_path = ROOT / 'artifacts' / 'distilbert' / 'progress.json'
    if progress_path.exists() and 'distilbert' not in available:
        try:
            progress = json.loads(progress_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            progress = None
        if progress:
            st.subheader('Latest recorded training progress')
            completed = (progress['epoch'] - 1) * progress['steps_per_epoch'] + progress['step']
            st.write(f"Epoch {progress['epoch']} · batch {progress['step']} of {progress['steps_per_epoch']} · elapsed {progress['elapsed_seconds'] / 60:.1f} minutes")
            st.caption('This is the last saved progress snapshot, not a live process-health check. Validation and final evaluation take additional time.')
            if st.button('Refresh training status'):
                st.rerun()
    report = read_report(model_name)
    if report.get('training_status'):
        st.info(report['training_status'])
    st.subheader('Measured model results')
    a, b = st.columns(2)
    a.metric('Short-text test macro-F1', f"{report['test']['macro_f1']:.3f}")
    b.metric('Short-text test accuracy', f"{report['test']['accuracy']:.1%}")
    st.caption('Evaluated on the original emotion dataset after duplicate controls. These are not real-journal performance estimates. The unknown-source data.zip is not used.')
    with st.expander('Evaluation details'):
        st.json(report)
    if comparison:
        st.subheader('Research comparison')
        table = [{
            'Model': name,
            'Validation macro-F1': values['benchmark_validation_macro_f1'],
            'Test macro-F1': values['benchmark_test_macro_f1'],
            'Test accuracy': values['benchmark_test_accuracy'],
            'Development macro-F1 (passages)': values['development_chunked_macro_f1'],
            'Development macro-F1 (whole)': values['development_whole_macro_f1'],
            'Median journal latency (ms)': values['median_journal_latency_ms'],
        } for name, values in comparison['models'].items()]
        st.dataframe(pd.DataFrame(table), hide_index=True)
        st.caption('Development cases are AI-authored diagnostics, not independent human-labelled evaluation. Neutral and mixed cases are assessed separately from single-label macro-F1.')
        with st.expander('Inspect development examples and failures'):
            values = comparison['models'].get(model_name)
            if values:
                st.write(f"Neutral cases returning uncertain: {values['neutral_abstained']}/{values['neutral_count']}. Mixed cases with both expected passage labels: {values['mixed_both_detected']}/{values['mixed_count']}.")
                st.dataframe(pd.DataFrame([{'Text': case['text'], 'Category': case['category'], 'Expected': ', '.join(case['expected_emotions']) or 'neutral / abstain', 'Prediction': case['prediction'], 'Passages': ', '.join(case['passage_labels'])} for case in values['cases']]), hide_index=True)

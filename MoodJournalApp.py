# moodjournal_app.py
# MoodJournal: NLP-Based Mood Trajectory Analysis for Mental Health Monitoring
# A standalone BiLSTM-based journaling tool for longitudinal mental health tracking.
#
# Key features of MoodJournal:
#   1. Free-form daily journal entry (not just a one-liner)
#   2. Full emotion probability breakdown (all 6 emotions, not just top-1)
#   3. SQLite-backed longitudinal store — persists entries across sessions
#   4. 7-day mood timeline chart
#   5. Trend alert — flags 3+ consecutive days of dominant negative emotion
#   6. Word-level importance highlight (poor-man's SHAP via per-word prediction delta)
#   7. Spotify recommendation by valence/energy audio features (not hardcoded URIs)
#   8. Coping suggestion (breathing / CBT prompt) for negative mood states
#
# ── How to run ────────────────────────────────────────────────────────────────
#   pip install streamlit spotipy tensorflow nltk scikit-learn pandas matplotlib
#   streamlit run moodjournal_app.py
#
# ── Files expected ───────────────────────────────────────────────────────────
#   MODEL_PATH        — .h5 keras model
#   TOKENIZER_PATH    — tokenizer.pkl
#   LABEL_ENC_PATH    — label_encoder.pkl
#   MAXLEN_PATH       — maxlen.pkl
#   PREPROCESSOR_PATH — text_preprocessor.pkl  (TextPreprocessor class)
#
# ── Spotify credentials ───────────────────────────────────────────────────────
#   Set CLIENT_ID, CLIENT_SECRET, REDIRECT_URI below.
# ─────────────────────────────────────────────────────────────────────────────

import os
import re
import string
import pickle
import sqlite3
import datetime
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.WARNING)

# ── Paths — update to match your environment ──────────────────────────────────
BASE_PATH         = Path('/content/drive/MyDrive/moodjournal')
MODEL_PATH        = BASE_PATH / 'bilstm_emotion_model.h5'
TOKENIZER_PATH    = BASE_PATH / 'tokenizer.pkl'
LABEL_ENC_PATH    = BASE_PATH / 'label_encoder.pkl'
MAXLEN_PATH       = BASE_PATH / 'maxlen.pkl'
PREPROCESSOR_PATH = BASE_PATH / 'text_preprocessor.pkl'
DB_PATH           = BASE_PATH / 'moodjournal.db'   # NEW: SQLite journal store

# ── Spotify credentials ───────────────────────────────────────────────────────
CLIENT_ID = os.environ.get('SPOTIFY_CLIENT_ID', '')
CLIENT_SECRET = os.environ.get('SPOTIFY_CLIENT_SECRET', '')
REDIRECT_URI  = 'http://localhost:5000/callback'
SCOPE         = 'user-modify-playback-state user-read-playback-state'

# ── Constants ─────────────────────────────────────────────────────────────────
EMOTIONS      = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']
NEGATIVE_EMOTIONS = {'anger', 'fear', 'sadness'}
ALERT_STREAK  = 3   # consecutive negative days before alert fires

# Valence + energy targets per emotion for Spotify audio-feature search
# (valence: 0=sad, 1=happy | energy: 0=calm, 1=intense)
EMOTION_AUDIO_TARGETS = {
    'anger':    {'valence': 0.3, 'energy': 0.8},
    'fear':     {'valence': 0.2, 'energy': 0.4},
    'joy':      {'valence': 0.9, 'energy': 0.8},
    'love':     {'valence': 0.8, 'energy': 0.5},
    'sadness':  {'valence': 0.3, 'energy': 0.3},
    'surprise': {'valence': 0.7, 'energy': 0.7},
}

# Coping suggestions per negative emotion
COPING = {
    'anger': [
        "🧘 Try box breathing: inhale 4s → hold 4s → exhale 4s → hold 4s. Repeat 4 times.",
        "✍️  Write down what triggered this feeling without judgment. Just observe.",
        "🚶 A 10-minute walk — no phone — can lower cortisol significantly.",
    ],
    'fear': [
        "🌬️ 4-7-8 breathing: inhale 4s, hold 7s, exhale 8s. Calms the nervous system.",
        "📝 List three things you can control right now. Focus only on those.",
        "🤝 Name the fear out loud or in writing — labelling reduces its intensity.",
    ],
    'sadness': [
        "🫂 Reach out to one person today — even a short message counts.",
        "🎵 The playlist below is tuned to gently lift your energy. Let it run in the background.",
        "🌿 Spend 5 minutes outside. Natural light measurably shifts mood.",
    ],
}

EMOTION_COLORS = {
    'anger':    '#D85A30',
    'fear':     '#7F77DD',
    'joy':      '#1D9E75',
    'love':     '#D4537E',
    'sadness':  '#378ADD',
    'surprise': '#EF9F27',
}

# ── NLTK downloads ────────────────────────────────────────────────────────────
@st.cache_resource
def download_nltk():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

download_nltk()

# ── TextPreprocessor ─────────────────────────────────────────────────────────
class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

    def lemmatization(self, text):
        text = text.split()
        text = [self.lemmatizer.lemmatize(word, pos='v') for word in text]
        return " ".join(text)

    def remove_stop_words(self, text):
        return " ".join([w for w in text.split() if w not in self.stop_words])

    def remove_numbers(self, text):
        return ''.join([c for c in text if not c.isdigit()])

    def lower_case(self, text):
        return text.lower()

    def remove_punctuations(self, text):
        text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def remove_urls(self, text):
        return re.sub(r'https?://\S+|www\.\S+', '', text)

    def normalized_sentence(self, sentence):
        sentence = self.lower_case(sentence)
        sentence = self.remove_stop_words(sentence)
        sentence = self.remove_numbers(sentence)
        sentence = self.remove_punctuations(sentence)
        sentence = self.remove_urls(sentence)
        sentence = self.lemmatization(sentence)
        return sentence


# ── Model & artefact loading ──────────────────────────────────────────────────
@st.cache_resource
def load_model_artefacts():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    with open(LABEL_ENC_PATH, 'rb') as f:
        le = pickle.load(f)
    with open(MAXLEN_PATH, 'rb') as f:
        maxlen = pickle.load(f)
    # Use pickled preprocessor if available, otherwise instantiate fresh
    if PREPROCESSOR_PATH.exists():
        with open(PREPROCESSOR_PATH, 'rb') as f:
            preprocessor = pickle.load(f)
    else:
        preprocessor = TextPreprocessor()
    return model, tokenizer, le, maxlen, preprocessor


# ── Prediction helpers ────────────────────────────────────────────────────────
def predict_emotions(text, model, tokenizer, maxlen, preprocessor):
    """Returns dict of {emotion: probability} for all 6 emotions."""
    cleaned = preprocessor.normalized_sentence(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=maxlen, truncating='pre')
    probs = model.predict(padded, verbose=0)[0]
    return {emotion: float(probs[i]) for i, emotion in enumerate(EMOTIONS)}


def top_emotion(prob_dict):
    return max(prob_dict, key=prob_dict.get)


def word_importance(text, model, tokenizer, maxlen, preprocessor):
    """
    Lightweight word attribution: for each word, measure how much the
    top-emotion probability drops when that word is masked (replaced with UNK).
    Returns list of (word, importance_score) sorted by importance desc.
    """
    base_probs = predict_emotions(text, model, tokenizer, maxlen, preprocessor)
    primary = top_emotion(base_probs)
    base_score = base_probs[primary]

    words = text.split()
    scores = []
    for i, word in enumerate(words):
        masked = words[:i] + [''] + words[i+1:]
        masked_text = ' '.join(masked)
        try:
            masked_probs = predict_emotions(masked_text, model, tokenizer, maxlen, preprocessor)
            delta = base_score - masked_probs[primary]
        except Exception:
            delta = 0.0
        scores.append((word, delta))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


# ── SQLite journal store ──────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS entries (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            date      TEXT NOT NULL,
            entry     TEXT NOT NULL,
            emotion   TEXT NOT NULL,
            anger     REAL, fear REAL, joy REAL,
            love      REAL, sadness REAL, surprise REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()


def save_entry(date_str, entry_text, prob_dict):
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        INSERT OR REPLACE INTO entries
          (date, entry, emotion, anger, fear, joy, love, sadness, surprise)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        date_str, entry_text, top_emotion(prob_dict),
        prob_dict['anger'], prob_dict['fear'], prob_dict['joy'],
        prob_dict['love'], prob_dict['sadness'], prob_dict['surprise'],
    ))
    conn.commit()
    conn.close()


def load_history(days=14):
    """Load the last N days of entries as a DataFrame."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM entries ORDER BY date DESC LIMIT ?",
        conn, params=(days,)
    )
    conn.close()
    return df.sort_values('date')


def detect_streak(history_df):
    """Returns True if the last ALERT_STREAK days are all negative emotions."""
    if len(history_df) < ALERT_STREAK:
        return False
    recent = history_df.tail(ALERT_STREAK)['emotion'].tolist()
    return all(e in NEGATIVE_EMOTIONS for e in recent)


# ── Spotify helpers ───────────────────────────────────────────────────────────
@st.cache_resource
def get_spotify():
    try:
        sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            redirect_uri=REDIRECT_URI,
            scope=SCOPE,
            open_browser=False,
            cache_path='.spotifycache',
        ))
        return sp
    except Exception:
        return None


def recommend_tracks(emotion, sp, n=3):
    """
    Search Spotify for tracks matching the emotion's audio-feature targets.
    Searches a genre seed matching the mood then filters by valence/energy.
    Falls back to a keyword search if recommendations API is unavailable.
    """
    if sp is None:
        return []

    targets = EMOTION_AUDIO_TARGETS[emotion]
    genre_map = {
        'anger':    'metal',
        'fear':     'ambient',
        'joy':      'pop',
        'love':     'romance',
        'sadness':  'acoustic',
        'surprise': 'indie',
    }
    try:
        results = sp.recommendations(
            seed_genres=[genre_map[emotion]],
            target_valence=targets['valence'],
            target_energy=targets['energy'],
            limit=n,
        )
        tracks = []
        for t in results['tracks']:
            tracks.append({
                'name':   t['name'],
                'artist': t['artists'][0]['name'],
                'uri':    t['uri'],
                'url':    t['external_urls']['spotify'],
            })
        return tracks
    except Exception:
        # Fallback: keyword search
        try:
            query_map = {
                'anger': 'calm down angry',
                'fear':  'peaceful ambient',
                'joy':   'happy upbeat',
                'love':  'love romantic',
                'sadness': 'healing sad acoustic',
                'surprise': 'uplifting indie',
            }
            r = sp.search(q=query_map[emotion], type='track', limit=n)
            return [{
                'name':   t['name'],
                'artist': t['artists'][0]['name'],
                'uri':    t['uri'],
                'url':    t['external_urls']['spotify'],
            } for t in r['tracks']['items']]
        except Exception:
            return []


def play_track(sp, uri):
    try:
        devices = sp.devices()
        if devices['devices']:
            device_id = devices['devices'][0]['id']
            sp.start_playback(device_id=device_id, uris=[uri])
            return True
    except Exception:
        pass
    return False


# ── Chart helpers ─────────────────────────────────────────────────────────────
def plot_timeline(history_df):
    if history_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 2.8))
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')

    dates = history_df['date'].tolist()
    emotions_list = history_df['emotion'].tolist()

    bar_colors = [EMOTION_COLORS.get(e, '#888') for e in emotions_list]
    # Use the primary-emotion probability as bar height
    heights = []
    for _, row in history_df.iterrows():
        heights.append(row[row['emotion']])

    x = range(len(dates))
    bars = ax.bar(x, heights, color=bar_colors, width=0.6, zorder=2)

    ax.set_xticks(list(x))
    ax.set_xticklabels([d[-5:] for d in dates], fontsize=9, color='#888')
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(['0', '0.5', '1.0'], fontsize=9, color='#888')
    ax.set_ylim(0, 1.1)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color('#ddd')
    ax.tick_params(axis='both', length=0)
    ax.set_ylabel('Confidence', fontsize=9, color='#888')

    # legend
    patches = [mpatches.Patch(color=EMOTION_COLORS[e], label=e) for e in EMOTIONS]
    ax.legend(handles=patches, fontsize=8, loc='upper left',
              frameon=False, ncol=3)

    plt.tight_layout()
    return fig


def plot_emotion_bars(prob_dict):
    fig, ax = plt.subplots(figsize=(5, 2.4))
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')

    sorted_emotions = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
    labels = [e for e, _ in sorted_emotions]
    values = [v for _, v in sorted_emotions]
    colors = [EMOTION_COLORS[e] for e in labels]

    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1], height=0.5, zorder=2)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel('Probability', fontsize=9, color='#888')
    ax.spines[['top', 'right', 'bottom']].set_visible(False)
    ax.spines['left'].set_color('#ddd')
    ax.tick_params(axis='both', length=0)
    ax.xaxis.set_tick_params(labelsize=9, labelcolor='#888')
    ax.yaxis.set_tick_params(labelsize=10, labelcolor='#444')

    for bar, val in zip(bars, values[::-1]):
        ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                f'{val:.0%}', va='center', fontsize=9, color='#888')
    plt.tight_layout()
    return fig


# ── Word highlight rendering ──────────────────────────────────────────────────
def render_highlighted_text(text, importance_scores):
    """
    Returns an HTML string with words coloured by importance.
    Top-3 contributing words get a highlighted background.
    """
    word_to_score = {w: s for w, s in importance_scores}
    top_words = set(w for w, s in importance_scores[:3] if s > 0)
    mid_words = set(w for w, s in importance_scores[3:8] if s > 0)

    html = ""
    for word in text.split():
        clean = word.lower().strip(string.punctuation)
        if clean in top_words:
            html += (f'<span style="background:#EEEDFE;color:#3C3489;'
                     f'border-radius:4px;padding:1px 5px;margin:1px;">{word}</span> ')
        elif clean in mid_words:
            html += (f'<span style="background:#E1F5EE;color:#085041;'
                     f'border-radius:4px;padding:1px 5px;margin:1px;">{word}</span> ')
        else:
            html += f'<span style="color:inherit;">{word}</span> '
    return html.strip()


# ── Streamlit UI ──────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="MoodJournal",
        page_icon="📓",
        layout="centered",
    )

    # ── DB init ───────────────────────────────────────────────────────────────
    init_db()

    # ── Load artefacts ────────────────────────────────────────────────────────
    try:
        model, tokenizer, le, maxlen, preprocessor = load_model_artefacts()
    except Exception as e:
        st.error(f"Could not load model artefacts: {e}")
        st.info("Make sure MODEL_PATH, TOKENIZER_PATH, LABEL_ENC_PATH, "
                "MAXLEN_PATH, and PREPROCESSOR_PATH are all set correctly at the top of this file.")
        st.stop()

    sp = get_spotify()

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("## 📓 MoodJournal")
    st.markdown(
        "<p style='color:#888;margin-top:-12px;'>NLP-Based Mood Trajectory Analysis "
        "for Mental Health Monitoring</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_journal, tab_history, tab_about = st.tabs(
        ["📝 Today's entry", "📈 Mood timeline", "ℹ️ About"]
    )

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — Today's entry
    # ════════════════════════════════════════════════════════════════════════
    with tab_journal:
        today = datetime.date.today().isoformat()
        st.markdown(f"**{datetime.date.today().strftime('%A, %d %B %Y')}**")

        entry = st.text_area(
            "How are you feeling today? Write freely.",
            height=160,
            placeholder="Woke up feeling a bit heavy... had a decent lunch though. "
                        "Work was stressful but managed to finish the report...",
        )

        col_save, col_clear = st.columns([3, 1])
        analyse_clicked = col_save.button("Analyse & save entry →", type="primary")

        if analyse_clicked:
            if not entry.strip():
                st.warning("Please write something before analysing.")
            else:
                with st.spinner("Analysing your entry…"):
                    prob_dict = predict_emotions(entry, model, tokenizer, maxlen, preprocessor)
                    primary = top_emotion(prob_dict)
                    save_entry(today, entry, prob_dict)

                # ── Alert banner ──────────────────────────────────────────
                history_df = load_history(days=14)
                if detect_streak(history_df):
                    st.warning(
                        f"⚠️ Your mood has been predominantly **{history_df.tail(1)['emotion'].values[0]}** "
                        f"for {ALERT_STREAK}+ days. Consider speaking to someone you trust, "
                        f"or reaching out to a mental health professional.",
                        icon="🔔",
                    )

                # ── Metric row ────────────────────────────────────────────
                c1, c2, c3 = st.columns(3)
                c1.metric("Primary emotion", primary.capitalize())
                c2.metric("Confidence", f"{prob_dict[primary]:.0%}")
                # Mood score: weighted sum leaning positive
                mood_score = round(
                    prob_dict['joy'] * 100
                    + prob_dict['love'] * 80
                    + prob_dict['surprise'] * 60
                    - prob_dict['sadness'] * 70
                    - prob_dict['anger'] * 80
                    - prob_dict['fear'] * 60,
                    1,
                )
                c3.metric("Mood score", f"{mood_score:+.0f}", help="Positive = uplifting emotions dominant")

                st.divider()

                # ── Emotion breakdown chart ───────────────────────────────
                st.markdown("#### Emotion breakdown")
                fig_bars = plot_emotion_bars(prob_dict)
                st.pyplot(fig_bars, use_container_width=True)

                # ── Word importance ───────────────────────────────────────
                st.markdown("#### Words driving this prediction")
                st.caption(
                    "Purple = strong signal · Green = moderate signal · "
                    "Based on word-masking attribution."
                )
                importance = word_importance(entry, model, tokenizer, maxlen, preprocessor)
                highlighted_html = render_highlighted_text(entry, importance)
                st.markdown(
                    f"<div style='line-height:2;font-size:15px;'>{highlighted_html}</div>",
                    unsafe_allow_html=True,
                )

                st.divider()

                # ── Coping suggestions (negative emotions only) ───────────
                if primary in COPING:
                    st.markdown("#### Suggested coping strategies")
                    for tip in COPING[primary]:
                        st.markdown(f"- {tip}")
                    st.divider()

                # ── Spotify recommendations ───────────────────────────────
                st.markdown("#### Recommended playlist")
                if sp is None:
                    st.info("Connect Spotify to get personalised track recommendations.")
                else:
                    tracks = recommend_tracks(primary, sp)
                    if tracks:
                        for t in tracks:
                            col_t, col_p = st.columns([5, 1])
                            col_t.markdown(
                                f"**{t['name']}** — *{t['artist']}*  \n"
                                f"[Open in Spotify]({t['url']})"
                            )
                            if col_p.button("▶ Play", key=t['uri']):
                                if play_track(sp, t['uri']):
                                    st.toast(f"Playing: {t['name']}")
                                else:
                                    st.warning("No active Spotify device found. "
                                               "Open Spotify on any device first.")
                    else:
                        st.info("Could not fetch tracks. Check Spotify credentials.")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — Mood timeline
    # ════════════════════════════════════════════════════════════════════════
    with tab_history:
        st.markdown("#### Your mood over the last 14 days")
        history_df = load_history(days=14)

        if history_df.empty:
            st.info("No entries yet. Start writing in the journal tab!")
        else:
            fig_timeline = plot_timeline(history_df)
            if fig_timeline:
                st.pyplot(fig_timeline, use_container_width=True)

            # Summary stats
            st.divider()
            st.markdown("#### Summary")
            emotion_counts = history_df['emotion'].value_counts()
            dominant = emotion_counts.index[0]
            negative_days = history_df[history_df['emotion'].isin(NEGATIVE_EMOTIONS)].shape[0]

            c1, c2, c3 = st.columns(3)
            c1.metric("Total entries", len(history_df))
            c2.metric("Dominant mood", dominant.capitalize())
            c3.metric("Days with negative mood", negative_days)

            st.divider()
            st.markdown("#### Entry history")
            display_df = history_df[['date', 'emotion', 'joy', 'sadness', 'anger', 'fear', 'love', 'surprise']].copy()
            display_df.columns = ['Date', 'Primary', 'Joy', 'Sadness', 'Anger', 'Fear', 'Love', 'Surprise']
            display_df = display_df.sort_values('Date', ascending=False)
            for col in ['Joy', 'Sadness', 'Anger', 'Fear', 'Love', 'Surprise']:
                display_df[col] = display_df[col].map(lambda x: f"{x:.0%}")
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            # PHQ-9 proxy note
            st.divider()
            st.markdown("#### Clinical proxy (research mode)")
            st.caption(
                "This is an experimental feature for the research paper. "
                "It maps NLP emotion scores to PHQ-9 item proxies — not a clinical diagnostic tool."
            )
            # PHQ-9 approximation: sadness + fear + anger averaged over entries
            avg_negative = history_df[['sadness', 'anger', 'fear']].mean().mean()
            phq_proxy = round(avg_negative * 27, 1)  # scale to PHQ-9 range 0–27
            severity = (
                "Minimal" if phq_proxy < 5 else
                "Mild" if phq_proxy < 10 else
                "Moderate" if phq_proxy < 15 else
                "Moderately severe" if phq_proxy < 20 else
                "Severe"
            )
            col_phq, col_sev = st.columns(2)
            col_phq.metric("PHQ-9 proxy score", f"{phq_proxy:.1f} / 27")
            col_sev.metric("Inferred severity", severity)
            st.caption(
                "⚠️ This proxy is derived solely from NLP emotion probabilities and is "
                "**not a substitute** for a clinician-administered PHQ-9."
            )

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3 — About
    # ════════════════════════════════════════════════════════════════════════
    with tab_about:
        st.markdown("""
### MoodJournal
**NLP-Based Mood Trajectory Analysis for Mental Health Monitoring**

A standalone Computational Intelligence minor project focused on passive, journal-driven mental health monitoring.

---

#### How it works

1. **You write freely** — no structured form, no scales to fill in.
2. **The BiLSTM model** (trained on the Kaggle emotions NLP dataset with GloVe 200d embeddings)
   classifies your text across 6 emotions: anger, fear, joy, love, sadness, surprise.
3. **All 6 probabilities are stored** per day in a local SQLite database.
4. **Word-level attribution** uses prediction-delta masking to highlight which words
   drove the classification — a lightweight alternative to SHAP.
5. **The timeline** plots your emotional trajectory over the last 14 days,
   flagging streaks of negative emotion.
6. **PHQ-9 proxy** (research feature) maps averaged NLP scores to the PHQ-9 severity scale
   for comparison against clinician-administered scores in a user study.
7. **Spotify recommendations** use the audio features API (valence, energy) to find tracks
   that match — or gently shift — your detected emotional state.

---

#### Research questions (paper)

- **RQ1:** How accurately does the BiLSTM classifier detect emotions from naturalistic
  journaling text compared to short-form social media text?
- **RQ2:** Does the NLP-derived emotion trajectory over 2–4 weeks correlate with
  pre/post changes in PHQ-9 and GAD-7 scores?
- **RQ3:** Which linguistic features are most predictive of sustained negative mood trajectories?

---

#### Dataset & model

| Component | Detail |
|---|---|
| Training data | Kaggle emotions-dataset-for-nlp (16k sentences, 6 classes) |
| Embeddings | GloVe 6B 200d |
| Architecture | Bidirectional LSTM × 2, Dense 64, Softmax 6 |
| Accuracy | >90% on held-out test set |
| Core features | Longitudinal store, word attribution, PHQ-9 proxy, audio-feature Spotify recommendations |

---
*MoodJournal is not a medical device. If you are in distress, please reach out to a mental health professional.*
        """)


if __name__ == "__main__":
    main()

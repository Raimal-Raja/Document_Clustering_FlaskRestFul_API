import streamlit as st
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Document Clustering",
    page_icon="🧠",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&display=swap');

/* ── global ── */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background: #0a0a0f;
    color: #e8e4dc;
}

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2.5rem; max-width: 720px; }

/* ── hero title ── */
.hero-title {
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -2px;
    line-height: 1;
    background: linear-gradient(135deg, #f0e6d0 30%, #c8a96e 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.25rem;
}
.hero-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 3px;
    color: #7a7060;
    text-transform: uppercase;
    margin-bottom: 2.5rem;
}

/* ── input area ── */
.stTextArea textarea {
    background: #111118 !important;
    border: 1px solid #2a2820 !important;
    border-radius: 12px !important;
    color: #e8e4dc !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.9rem !important;
    padding: 1rem !important;
    transition: border-color 0.2s;
}
.stTextArea textarea:focus {
    border-color: #c8a96e !important;
    box-shadow: 0 0 0 2px rgba(200,169,110,0.15) !important;
}

/* ── button ── */
.stButton > button {
    background: #c8a96e !important;
    color: #0a0a0f !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    padding: 0.65rem 2rem !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* ── result card ── */
.result-card {
    background: linear-gradient(135deg, #111118, #16141a);
    border: 1px solid #2a2820;
    border-left: 4px solid #c8a96e;
    border-radius: 12px;
    padding: 1.5rem 1.75rem;
    margin-top: 1.5rem;
}
.result-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 3px;
    color: #7a7060;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.result-value {
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -1px;
    color: #c8a96e;
}
.result-icon { font-size: 2rem; float: right; margin-top: -0.5rem; }

/* ── categories sidebar ── */
.cat-pill {
    display: inline-block;
    background: #111118;
    border: 1px solid #1e1d24;
    border-radius: 20px;
    padding: 0.2rem 0.75rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #7a7060;
    margin: 0.2rem 0.2rem 0.2rem 0;
}

/* ── divider ── */
hr { border-color: #1e1d24 !important; }

/* ── stats row ── */
.stat-box {
    background: #111118;
    border: 1px solid #1e1d24;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.stat-num {
    font-size: 1.8rem;
    font-weight: 800;
    color: #c8a96e;
    letter-spacing: -1px;
}
.stat-lbl {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 2px;
    color: #4a4840;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)

# ── Dataset & category metadata ───────────────────────────────────────────────
CATEGORY_ICONS = {
    "Technology":    "💻",
    "Sports":        "⚽",
    "Education":     "📚",
    "Health":        "🏥",
    "Environment":   "🌿",
    "Business":      "💼",
    "Space":         "🚀",
    "Travel":        "✈️",
    "Food":          "🍽️",
}

EXTENDED_DATASET = [
    # Technology / AI / ML
    ("Artificial intelligence is transforming modern technology.", "Technology"),
    ("Machine learning algorithms improve data analysis.", "Technology"),
    ("Deep learning models are trained on large datasets.", "Technology"),
    ("Neural networks mimic the human brain structure.", "Technology"),
    ("Natural language processing helps computers understand text.", "Technology"),
    ("Cloud computing helps store large amounts of data.", "Technology"),
    ("Cybersecurity protects computer systems from attacks.", "Technology"),
    ("Online learning platforms are growing rapidly.", "Technology"),
    ("Mobile phones have become essential communication tools.", "Technology"),
    ("Laptops and tablets are widely used for work and study.", "Technology"),
    ("Electronic devices improve productivity in daily life.", "Technology"),

    # Sports
    ("Cricket is the most popular sport in Pakistan.", "Sports"),
    ("Football is played worldwide by millions of fans.", "Sports"),
    ("The Olympics include many international sports competitions.", "Sports"),
    ("Players train hard to improve their performance.", "Sports"),
    ("Basketball teams compete in national and international leagues.", "Sports"),
    ("Tennis tournaments attract top athletes from around the world.", "Sports"),
    ("Coaches develop strategies to help teams win matches.", "Sports"),

    # Education
    ("Universities provide higher education and research opportunities.", "Education"),
    ("Students attend lectures and complete assignments regularly.", "Education"),
    ("Teachers help students understand complex subjects.", "Education"),
    ("E-learning systems allow students to study from home.", "Education"),
    ("Online education makes learning accessible globally.", "Education"),
    ("Digital classrooms use technology to enhance teaching.", "Education"),

    # Health
    ("Regular exercise improves overall health and fitness.", "Health"),
    ("Doctors recommend balanced diets for a healthy lifestyle.", "Health"),
    ("Hospitals provide medical treatment for various diseases.", "Health"),
    ("Vaccines prevent many dangerous and infectious diseases.", "Health"),
    ("Medical research focuses on finding new treatments.", "Health"),
    ("Healthcare systems support patients in recovery.", "Health"),

    # Environment
    ("Climate change is causing a rise in global temperatures.", "Environment"),
    ("Trees and forests help reduce air pollution levels.", "Environment"),
    ("Recycling materials helps protect the natural environment.", "Environment"),
    ("Renewable energy sources reduce carbon emissions significantly.", "Environment"),
    ("Environmental protection policies aim to preserve biodiversity.", "Environment"),
    ("Ocean pollution is a growing threat to marine life.", "Environment"),

    # Business
    ("Entrepreneurs build startups to solve real-world problems.", "Business"),
    ("Business strategies help companies achieve their financial goals.", "Business"),
    ("Marketing campaigns increase brand visibility and sales.", "Business"),
    ("Investors fund promising startups to scale their operations.", "Business"),
    ("Supply chain management is critical for business efficiency.", "Business"),

    # Space
    ("Space exploration involves sending probes to distant planets.", "Space"),
    ("Satellites orbit Earth and provide GPS navigation signals.", "Space"),
    ("Astronauts conduct scientific experiments aboard the ISS.", "Space"),
    ("Rocket launches are carefully planned to reach outer space.", "Space"),
    ("Space agencies research the possibility of life on Mars.", "Space"),

    # Travel
    ("Tourism promotes cultural exchange between different nations.", "Travel"),
    ("Travelers explore famous destinations and historic landmarks.", "Travel"),
    ("Hotels offer comfortable accommodation for tourists worldwide.", "Travel"),
    ("Travel agencies plan itineraries for vacation trips abroad.", "Travel"),
    ("Adventure travel includes hiking, diving, and mountaineering.", "Travel"),

    # Food
    ("Restaurants serve a variety of cuisines from around the world.", "Food"),
    ("Cooking recipes teach people to prepare delicious meals.", "Food"),
    ("Food festivals celebrate local culinary traditions and culture.", "Food"),
    ("Chefs experiment with ingredients to create new dishes.", "Food"),
    ("Healthy eating habits reduce the risk of chronic diseases.", "Food"),
]

# ── DB & model (cached) ───────────────────────────────────────────────────────
import os
DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "text_classifier.db")

@st.cache_resource
def init_and_train():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS Documents (
            DocID INTEGER PRIMARY KEY AUTOINCREMENT,
            Content TEXT,
            Label VARCHAR(50)
        )
    """)
    c.execute("SELECT COUNT(*) FROM Documents")
    if c.fetchone()[0] == 0:
        c.executemany("INSERT INTO Documents (Content, Label) VALUES (?, ?)", EXTENDED_DATASET)
        conn.commit()

    c.execute("SELECT Content, Label FROM Documents")
    rows = c.fetchall()
    conn.close()

    texts  = [r[0] for r in rows]
    labels = [r[1] for r in rows]

    vec = TfidfVectorizer()
    clf = MultinomialNB()
    X   = vec.fit_transform(texts)
    clf.fit(X, labels)

    return vec, clf, len(rows), len(set(labels))

vectorizer, classifier, n_docs, n_cats = init_and_train()

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">Document Clustering</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Naive Bayes · TF-IDF · SQLite</div>', unsafe_allow_html=True)

# Stats row
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f'<div class="stat-box"><div class="stat-num">{n_docs}</div><div class="stat-lbl">Training Docs</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="stat-box"><div class="stat-num">{n_cats}</div><div class="stat-lbl">Categories</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="stat-box"><div class="stat-num">NB</div><div class="stat-lbl">Algorithm</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Categories
pills = " ".join(
    f'<span class="cat-pill">{icon} {cat}</span>'
    for cat, icon in CATEGORY_ICONS.items()
)
st.markdown(f"<div style='margin-bottom:1.5rem'>{pills}</div>", unsafe_allow_html=True)

# Input
user_text = st.text_area(
    "Enter text to classify",
    placeholder="Paste or type any document here…",
    height=150,
    label_visibility="collapsed",
)

# Classify button
if st.button("CLASSIFY →"):
    text = user_text.strip()
    if not text:
        st.warning("Please enter some text first.")
    else:
        features = vectorizer.transform([text])
        prediction = classifier.predict(features)[0]
        icon = CATEGORY_ICONS.get(prediction, "📄")

        st.markdown(f"""
        <div class="result-card">
            <div class="result-icon">{icon}</div>
            <div class="result-label">Predicted Category</div>
            <div class="result-value">{prediction}</div>
        </div>
        """, unsafe_allow_html=True)

        # Probability bar
        proba = classifier.predict_proba(features)[0]
        classes = classifier.classes_
        top5 = sorted(zip(classes, proba), key=lambda x: -x[1])[:5]

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p style="font-family:\'DM Mono\',monospace;font-size:0.7rem;letter-spacing:2px;color:#4a4840;text-transform:uppercase">Confidence breakdown</p>', unsafe_allow_html=True)
        for label, prob in top5:
            lbl_icon = CATEGORY_ICONS.get(label, "📄")
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:0.5rem">
                <span style="font-size:1rem;width:24px">{lbl_icon}</span>
                <span style="font-family:'Syne',sans-serif;font-size:0.82rem;width:110px;color:#e8e4dc">{label}</span>
                <div style="flex:1;background:#1a1a22;border-radius:4px;height:6px;overflow:hidden">
                    <div style="width:{prob*100:.1f}%;background:#c8a96e;height:100%;border-radius:4px"></div>
                </div>
                <span style="font-family:'DM Mono',monospace;font-size:0.75rem;color:#7a7060;width:46px;text-align:right">{prob*100:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown('<p style="font-family:\'DM Mono\',monospace;font-size:0.65rem;letter-spacing:2px;color:#2a2820;text-align:center">SKLEARN · STREAMLIT · SQLITE</p>', unsafe_allow_html=True)
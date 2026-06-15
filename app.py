"""
RestoPulse v4.0 - Streamlit App
1 yulduzdan 5 yulduzgacha sentiment, aspekt tahlili, diagrammalar
"""

import os, json, pickle, re
import streamlit as st
import pandas as pd
import numpy as np

# Page configuration MUST be the first streamlit call
if 'lang_select' not in st.session_state:
    st.session_state['lang_select'] = 'English'

# Initialize language settings before set_page_config
is_eng_init = (st.session_state.get('lang_select', 'English') == "English")
page_title_init = "RestoPulse – AI Review Analysis" if is_eng_init else "RestoPulse – AI Sharh Tahlili"
st.set_page_config(page_title=page_title_init, page_icon="🍽️", layout="centered")

# Language Selector in Sidebar
lang = st.sidebar.selectbox("Language / Til", ["English", "O'zbekcha"], key="lang_select")
is_eng = (lang == "English")

from train_model import (
    predict, load_model, ASPECT_KEYWORDS,
    SENTIMENT_LABELS, SENTIMENT_COLORS, SENTIMENT_EMOJI, stars_display
)

# ─────────────────────────────────────────
# BILINGUAL UI TRANSLATIONS
# ─────────────────────────────────────────
UI_STRINGS = {
    "English": {
        "title_sub": "Uzbek Restaurant Reviews · ⭐ 1 to 5 Star Rating System",
        "tab_analysis": "🔍 Analysis",
        "tab_stats": "📊 Statistics",
        "tab_project": "ℹ️ Project Info",
        "samples_label": "⚡ Examples (in Uzbek):",
        "sample_terrible": "⭐ 1 Terrible",
        "sample_average": "⭐⭐⭐ Average",
        "sample_excellent": "⭐⭐⭐⭐⭐ Perfect",
        "input_header": "Enter review in Uzbek",
        "input_placeholder": "e.g., Taomlar mazali edi, lekin xizmat juda sekin bo'ldi... (The food was delicious, but the service was very slow...)",
        "btn_analyze": "🔎 Analyze Feedback",
        "warn_empty": "⚠️ Please enter a review!",
        "spinner_analyzing": "Analyzing feedback...",
        "confidence": "Confidence",
        "star_prob": "Star Rating Probability:",
        "xaxis_stars": "Star Rating",
        "yaxis_prob": "Probability %",
        "aspects_title": "🎯 Which aspects were mentioned",
        "aspects_none": "📝 No specific aspect detected",
        "aspects_general": "General feedback",
        "aspect_sentiment_title": "Customer sentiment by aspect (Negative / Positive):",
        "xaxis_sentiment": "Negative (Left) <---> Positive (Right)",
        "yaxis_aspects": "Aspect",
        "complaints_title": "⚠️ Complaints (What's wrong)",
        "praise_title": "🌟 Praises (What's good)",
        "no_complaints": "✔️ No complaints found",
        "no_praise": "📝 No praises found",
        "preprocessing_title": "🔬 Preprocessing & Cleaning",
        "csv_section_title": "📂 CSV Batch Analysis",
        "csv_info": "The CSV file must contain a <b>Review_Text</b> column.",
        "csv_uploader_label": "Upload CSV file",
        "csv_err_column": "❌ 'Review_Text' column not found!",
        "metric_total": "Total",
        "metric_average": "Average",
        "chart_star_dist": "Star Rating Distribution",
        "chart_share_by_level": "Share by Level",
        "chart_complaints_by_aspect": "Complaints — by aspect",
        "chart_praises_by_aspect": "Praises — by aspect",
        "table_title": "📋 Results Table",
        "filter_label": "Filter:",
        "btn_download": "⬇️ Download results",
        "tab_stats_metrics": "📊 Model Metrics",
        "tab_stats_exact_acc": "Exact Accuracy",
        "tab_stats_tolerance": "±1 star tolerance",
        "tab_stats_mae": "Mean Absolute Error (MAE)",
        "tab_stats_dist": "🗂 Dataset Distribution",
        "tab_stats_info": "ℹ️ Model Information",
        "tab_stats_col_metric": "Metric",
        "tab_stats_col_val": "Value",
        "tab_stats_info_train": "Training set",
        "tab_stats_info_test": "Test set",
        "tab_stats_info_vocab": "Vocabulary size",
        "tab_stats_info_model": "Model type",
        "tab_stats_info_scale": "Scale",
        "tab_stats_info_reviews": "reviews",
        "tab_stats_info_tokens": "tokens (TF-IDF 1-3 gram)",
        "tab_stats_not_trained": "❌ Model not found! Please run `python train_model.py` first.",
        "tab_project_intro": """
**Key Features (v4.0 — Star System):**
- ⭐ **1 to 5 Star Rating System** (simple and clear sentiment)
- 📊 **Star Distribution Charts** — visualize reviews by star levels
- 🎯 **Aspect Analysis** — 6 key areas: Food, Service, Price, Atmosphere, Cleanliness, Location
- 🤖 **Logistic Regression Model** — optimized for 5-class classification
- 📈 **TF-IDF 1-3 gram** text vectorization
- ±1 star tolerance accuracy indicator
        """,
        "tab_project_heading_stars": "Star Sentiment System:",
        "tab_project_heading_tech": "Technologies:",
        "tab_project_footer": "**👤 Student:** Toshmurodov Jumabek | **Unit 21** | **BTEC Level 3**",
        "sample_terrible_txt": "Ovqatdan zaharlandim, kasalxonaga tushdim. Eng dahshatli joy, hech kimga tavsiya qilmayman",
        "sample_average_txt": "Oddiy restoran, taom ham xizmat ham o'rtacha edi, na yaxshi na yomon",
        "sample_excellent_txt": "Hayotimda bunday mazali taom yemagandim, xizmat professional, muhit mukammal edi",
        "all_filter": "All",
        "aspect_positive": "Positive",
        "aspect_negative": "Negative",
        "info_box_csv_col": "CSV file must contain a <b>Review_Text</b> column.",
        "star_unit": "Star",
        "yulduz_label": "Stars",
        "daraja_label": "Level",
        "ishonch_label": "Confidence",
        "aspektlar_label": "Aspects",
        "result_file_name": "results_v4.csv",
        "aspects_detected_title": "🎯 Which aspects were mentioned",
    },
    "O'zbekcha": {
        "title_sub": "O'zbekcha restoran sharhlari · ⭐ 1 dan 5 yulduzgacha baho tizimi",
        "tab_analysis": "🔍 Tahlil",
        "tab_stats": "📊 Statistika",
        "tab_project": "ℹ️ Loyiha",
        "samples_label": "⚡ Namunalar:",
        "sample_terrible": "⭐ 1 Dahshatli",
        "sample_average": "⭐⭐⭐ O'rtacha",
        "sample_excellent": "⭐⭐⭐⭐⭐ Mukammal",
        "input_header": "Sharh kiriting",
        "input_placeholder": "Masalan: Taomlar mazali edi, lekin xizmat juda sekin bo'ldi...",
        "btn_analyze": "🔎 Tahlil qilish",
        "warn_empty": "⚠️ Iltimos, sharh kiriting!",
        "spinner_analyzing": "Tahlil qilinmoqda...",
        "confidence": "Ishonch",
        "star_prob": "Yulduz ehtimolligi:",
        "xaxis_stars": "Yulduz darajasi",
        "yaxis_prob": "%",
        "aspects_title": "🎯 Qaysi sohalar tilga olingan",
        "aspects_none": "📝 Aniq soha aniqlanmadi",
        "aspects_general": "Umumiy sharh",
        "aspect_sentiment_title": "Aspektlar bo'yicha mijoz munosabati (Salbiy / Ijobiy):",
        "xaxis_sentiment": "Salbiy (Chapda) <---> Ijobiy (O'ngda)",
        "yaxis_aspects": "Soha",
        "complaints_title": "⚠️ Nimadan norozi",
        "praise_title": "🌟 Nima maqtovga sazovor",
        "no_complaints": "✔️ Shikoyat topilmadi",
        "no_praise": "📝 Maqtov topilmadi",
        "preprocessing_title": "🔬 Preprocessing",
        "csv_section_title": "📂 CSV ommaviy tahlil",
        "csv_info": "CSV faylda <b>Review_Text</b> ustuni bo'lishi kerak.",
        "csv_uploader_label": "CSV fayl yuklang",
        "csv_err_column": "❌ 'Review_Text' ustuni topilmadi!",
        "metric_total": "Jami",
        "metric_average": "O'rtacha",
        "chart_star_dist": "Yulduz taqsimoti",
        "chart_share_by_level": "Daraja bo'yicha ulush",
        "chart_complaints_by_aspect": "Shikoyatlar — soha bo'yicha",
        "chart_praises_by_aspect": "Maqtovlar — soha bo'yicha",
        "table_title": "📋 Natijalar jadvali",
        "filter_label": "Filter:",
        "btn_download": "⬇️ Natijani yuklab olish",
        "tab_stats_metrics": "📊 Model Ko'rsatkichlari",
        "tab_stats_exact_acc": "Exact Accuracy",
        "tab_stats_tolerance": "±1 yulduz aniqlik",
        "tab_stats_mae": "O'rtacha xato (MAE)",
        "tab_stats_dist": "🗂 Dataset taqsimoti",
        "tab_stats_info": "ℹ️ Model ma'lumotlari",
        "tab_stats_col_metric": "Ko'rsatkich",
        "tab_stats_col_val": "Qiymat",
        "tab_stats_info_train": "O'qitish",
        "tab_stats_info_test": "Sinov",
        "tab_stats_info_vocab": "Lug'at",
        "tab_stats_info_model": "Model",
        "tab_stats_info_scale": "Shkala",
        "tab_stats_info_reviews": "sharh",
        "tab_stats_info_tokens": "token (TF-IDF 1-3 gram)",
        "tab_stats_not_trained": "❌ Model topilmadi! Avval `python train_model.py` ni ishga tushiring.",
        "tab_project_intro": """
**Yangiliklar (v4.0 — Yulduz tizimi):**
- ⭐ **1 dan 5 yulduzgacha baho tizimi** (oddiy va tushunarli)
- 📊 **Yulduz taqsimoti diagrammasi** — har daraja alohida ko'rsatiladi
- 🎯 **Aspekt tahlili** — 6 soha: Taom, Xizmat, Narx, Muhit, Tozalik, Joylashuv
- 🤖 **Logistic Regression** — 5 klass uchun optimallashtirilgan
- 📈 **TF-IDF 1-3 gram** vektorizatsiya
- ±1 yulduz aniqlik ko'rsatkichi
        """,
        "tab_project_heading_stars": "Yulduz tizimi:",
        "tab_project_heading_tech": "Texnologiyalar:",
        "tab_project_footer": "**👤 Talaba:** Toshmurodov Jumabek | **Unit 21** | **BTEC Level 3**",
        "sample_terrible_txt": "Ovqatdan zaharlandim, kasalxonaga tushdim. Eng dahshatli joy, hech kimga tavsiya qilmayman",
        "sample_average_txt": "Oddiy restoran, taom ham xizmat ham o'rtacha edi, na yaxshi na yomon",
        "sample_excellent_txt": "Hayotimda bunday mazali taom yemagandim, xizmat professional, muhit mukammal edi",
        "all_filter": "Barchasi",
        "aspect_positive": "Ijobiy",
        "aspect_negative": "Salbiy",
        "info_box_csv_col": "CSV faylda <b>Review_Text</b> ustuni bo'lishi kerak.",
        "star_unit": "Yulduz",
        "yulduz_label": "Yulduzlar",
        "daraja_label": "Daraja",
        "ishonch_label": "Ishonch",
        "aspektlar_label": "Aspektlar",
        "result_file_name": "natijalar_v4.csv",
        "aspects_detected_title": "🎯 Qaysi sohalar tilga olingan",
    }
}

def t(key):
    return UI_STRINGS[lang].get(key, key)

# Sentiment label overrides for English
if is_eng:
    UI_SENTIMENT_LABELS = {
        1: "Terrible",
        2: "Bad",
        3: "Average",
        4: "Good",
        5: "Excellent",
    }
else:
    UI_SENTIMENT_LABELS = SENTIMENT_LABELS

# Aspect labels translation
ASPECT_META_UZ = {
    "taom":      {"icon": "🍽️", "label": "Taom sifati"},
    "xizmat":    {"icon": "👨🍳", "label": "Xizmat sifati"},
    "narx":      {"icon": "💰",  "label": "Narx / Qiymat"},
    "muhit":     {"icon": "🏮",  "label": "Muhit / Atmosfera"},
    "tozalik":   {"icon": "✨",  "label": "Tozalik"},
    "joylashuv": {"icon": "📍",  "label": "Joylashuv"},
}

ASPECT_META_EN = {
    "taom":      {"icon": "🍽️", "label": "Food Quality"},
    "xizmat":    {"icon": "👨🍳", "label": "Service Quality"},
    "narx":      {"icon": "💰",  "label": "Price / Value"},
    "muhit":     {"icon": "🏮",  "label": "Atmosphere / Vibe"},
    "tozalik":   {"icon": "✨",  "label": "Cleanliness"},
    "joylashuv": {"icon": "📍",  "label": "Location"},
}

ASPECT_META = ASPECT_META_EN if is_eng else ASPECT_META_UZ

WORD_MAP = {
    # TAOM
    "taom":("taom","n"), "ovqat":("taom","n"), "shashlik":("taom","n"),
    "osh":("taom","n"), "lag'mon":("taom","n"), "kabob":("taom","n"),
    "non":("taom","n"), "salat":("taom","n"), "porsiya":("taom","n"),
    "mazali emas":("taom","neg"), "lazzatli emas":("taom","neg"), "ta'mli emas":("taom","neg"),
    "yangi emas":("taom","neg"), "fresh emas":("taom","neg"), "shirin emas":("taom","neg"),
    "mazali":("taom","p"), "lazzatli":("taom","p"), "ta'mli":("taom","p"),
    "yangi edi":("taom","p"), "fresh edi":("taom","p"), "shirin":("taom","p"), "shirin edi":("taom","p"),
    "ta'msiz":("taom","neg"), "sho'r edi":("taom","neg"),
    "achchiq edi":("taom","neg"), "chala edi":("taom","neg"),
    "sovuq keldi":("taom","neg"), "bayiy edi":("taom","neg"),
    "sifatsiz edi":("taom","neg"), "yoqimsiz edi":("taom","neg"),
    "kuygan":("taom","neg"), "achigan":("taom","neg"),
    # XIZMAT
    "xizmat":("xizmat","n"), "ofitsiant":("xizmat","n"),
    "xodim":("xizmat","n"), "buyurtma":("xizmat","n"),
    "muloyim emas":("xizmat","neg"), "iltifotli emas":("xizmat","neg"),
    "e'tiborli emas":("xizmat","neg"), "tez emas":("xizmat","neg"),
    "muloyim":("xizmat","p"), "iltifotli":("xizmat","p"),
    "professional":("xizmat","p"), "tez bo'ldi":("xizmat","p"),
    "e'tiborli":("xizmat","p"), "yaxshi xizmat":("xizmat","p"),
    "tezkor":("xizmat","p"), "xushmuomala":("xizmat","p"),
    "chaqqon":("xizmat","p"), "odobli":("xizmat","p"),
    "sekin":("xizmat","neg"), "kech keldi":("xizmat","neg"),
    "e'tibor bermadi":("xizmat","neg"), "unutdi":("xizmat","neg"),
    "qo'pol":("xizmat","neg"), "bilmayman dedi":("xizmat","neg"),
    "kutdim":("xizmat","neg"), "professional emas":("xizmat","neg"),
    # NARX
    "narx":("narx","n"), "narxi":("narx","n"), "pul":("narx","n"),
    "arzon emas":("narx","neg"), "adolatli emas":("narx","neg"),
    "chegirma":("narx","p"), "bepul":("narx","p"), "arzon":("narx","p"),
    "tejamkor":("narx","p"), "narxi qulay":("narx","p"), "adolatli":("narx","p"),
    "qimmat":("narx","neg"), "juda qimmat":("narx","neg"),
    "arzimaydi":("narx","neg"), "narxi baland":("narx","neg"),
    # MUHIT
    "muhit":("muhit","n"), "atmosfera":("muhit","n"),
    "bezak":("muhit","n"), "dizayn":("muhit","n"), "interior":("muhit","n"),
    "chiroyli emas":("muhit","neg"), "romantik emas":("muhit","neg"),
    "qulay emas":("muhit","neg"), "tinch emas":("muhit","neg"),
    "chiroyli":("muhit","p"), "romantik":("muhit","p"),
    "qulay muhit":("muhit","p"), "tinch":("muhit","p"),
    "ajoyib muhit":("muhit","p"), "atmosfera ajoyib":("muhit","p"),
    "shinam":("muhit","p"),
    "shovqin":("muhit","neg"), "tor":("muhit","neg"),
    "noqulay":("muhit","neg"), "noqulay muhit":("muhit","neg"), "eskirgan":("muhit","neg"),
    # TOZALIK
    "toza emas":("tozalik","neg"),
    "toza":("tozalik","p"), "tozalik":("tozalik","n"),
    "iflos":("tozalik","neg"), "yomon hid":("tozalik","neg"),
    "chang":("tozalik","neg"), "tozalanmagan":("tozalik","neg"),
    "soch":("tozalik","neg"), "suvarak":("tozalik","neg"),
    "chuvalchang":("tozalik","neg"), "chibin":("tozalik","neg"),
    "pashsha":("tozalik","neg"), "sichqon":("tozalik","neg"),
    "kalamush":("tozalik","neg"), "sassiq":("tozalik","neg"),
    # JOYLASHUV
    "manzil":("joylashuv","n"), "joylashuv":("joylashuv","n"),
    "parking":("joylashuv","n"), "yaqin":("joylashuv","p"),
    "topish oson":("joylashuv","p"), "uzoq":("joylashuv","neg"),
    "parking yo'q":("joylashuv","neg"), "topish qiyin":("joylashuv","neg"),
    "yaqin emas":("joylashuv","neg"), "topish oson emas":("joylashuv","neg"),
    # UMUMIY
    "halol":("umumiy","p"), "super":("umumiy","p"), "klass":("umumiy","p"), "bomba":("umumiy","p"),
    "aldash":("umumiy","neg"), "aldashdi":("umumiy","neg"), "kasal":("umumiy","neg"),
    "norozi":("umumiy","neg"), "sifatsiz":("umumiy","neg"), "yoqimsiz":("umumiy","neg"),
}


def detect_aspects(text: str, score: int = None):
    if score is None:
        try:
            score, _, _, _, _ = predict(text, model, vectorizer)
        except Exception:
            score = 3  # default if model not loaded yet
            
    # Preprocess text
    from train_model import preprocess_ml
    clean = preprocess_ml(text)
    if not clean.strip():
        return {}, [], []
    vec = vectorizer.transform([clean])
    
    found_aspects = {}
    complaints = []
    praises = []
    
    for asp, clf in aspect_models.items():
        if clf is not None:
            pred = clf.predict(vec)[0]
            if pred == 1:
                # Aspect detected! Find matching words from ASPECT_KEYWORDS for UI display
                matched_words = []
                text_lower = text.lower()
                for char in ["’", "‘", "ʻ", "ʼ", "´", "`"]:
                    text_lower = text_lower.replace(char, "'")
                
                # Match keywords starting at word boundaries (allowing suffixes, e.g. "taom" matches "taomlari")
                for kw in ASPECT_KEYWORDS.get(asp, []):
                    pattern = rf"(?<![a-zA-Z']){re.escape(kw)}"
                    if re.search(pattern, text_lower):
                        matched_words.append(kw)
                
                display_word = matched_words[0] if matched_words else ASPECT_META.get(asp, {"label": asp})["label"].lower()
                found_aspects[asp] = matched_words if matched_words else [display_word]
                
                meta = ASPECT_META.get(asp, {"icon": "💬", "label": asp})
                
                if score >= 4:
                    praises.append({"aspect": meta["label"], "icon": meta["icon"], "word": display_word})
                elif score <= 2:
                    complaints.append({"aspect": meta["label"], "icon": meta["icon"], "word": display_word})
                else:
                    # 3 stars: check if we match negative keywords
                    is_neg = False
                    for w in matched_words:
                        if w in WORD_MAP and WORD_MAP[w][1] == "neg":
                            is_neg = True
                            break
                    if is_neg:
                        complaints.append({"aspect": meta["label"], "icon": meta["icon"], "word": display_word})
                    else:
                        praises.append({"aspect": meta["label"], "icon": meta["icon"], "word": display_word})
                        
    return found_aspects, complaints, praises


def predict_aspect_probabilities(text: str):
    from train_model import preprocess_ml
    clean = preprocess_ml(text)
    if not clean.strip():
        return {asp: 0.0 for asp in aspect_models.keys()}
    vec = vectorizer.transform([clean])
    
    probs = {}
    for asp, clf in aspect_models.items():
        if clf is not None:
            probs[asp] = float(clf.predict_proba(vec)[0][1])
        else:
            probs[asp] = 0.0
    return probs


def score_to_group(score: int) -> str:
    if is_eng:
        if score == 5: return "Excellent"
        if score == 4: return "Good"
        if score == 3: return "Average"
        if score == 2: return "Bad"
        return "Terrible"
    else:
        if score == 5: return "A'lo"
        if score == 4: return "Yaxshi"
        if score == 3: return "O'rtacha"
        if score == 2: return "Yomon"
        return "Dahshatli"


# ─────────────────────────────────────────
# SAHIFA
# ─────────────────────────────────────────

@st.cache_resource
def get_model():
    if not os.path.exists("model/model.pkl"):
        st.error(t("tab_stats_not_trained"))
        st.stop()
    main_model, vectorizer = load_model("model")
    
    # Load aspect models
    aspect_models = {}
    aspects = ["taom", "xizmat", "narx", "muhit", "tozalik", "joylashuv"]
    for asp in aspects:
        path = f"model/model_{asp}.pkl"
        if os.path.exists(path):
            with open(path, "rb") as f:
                aspect_models[asp] = pickle.load(f)
        else:
            aspect_models[asp] = None
    return main_model, vectorizer, aspect_models

model, vectorizer, aspect_models = get_model()
metrics = {}
if os.path.exists("model/metrics.json"):
    with open("model/metrics.json") as f:
        metrics = json.load(f)

# ─── CSS ───
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Nunito', sans-serif !important; }

.title-box {
    background: linear-gradient(135deg, #ff6b35, #f7931e);
    border-radius: 20px; padding: 32px 36px; margin-bottom: 24px;
    text-align: center; color: #fff !important;
    box-shadow: 0 8px 32px rgba(255,107,53,.35);
}
.title-box h1 { font-size:2.2rem; margin:0; font-weight:800; color:#fff !important; }
.title-box p  { font-size:1rem; margin:8px 0 0; opacity:.95; color:#fff !important; }

.score-box {
    border-radius: 18px; padding: 24px 28px; text-align: center;
    margin: 16px 0; border: 2px solid;
}
.score-box .stars { font-size:2.8rem; line-height:1; letter-spacing:4px; }
.score-box .big   { font-size:2rem; font-weight:900; line-height:1.2; }
.score-box .label { font-size:1.3rem; font-weight:800; margin-top:6px; }
.score-box .conf  { font-size:0.82rem; opacity:.8; margin-top:4px; }

.aspect-card {
    background:#1e2535; border-radius:12px; padding:13px 16px; margin:5px 0;
    border-left:4px solid #ff6b35; color:#e8eaf0 !important; font-size:.9rem;
}
.aspect-card b { color:#ff9a6c !important; }
.aspect-card small { color:#9ba3b5 !important; }
.complaint-card {
    background:#2d1518; border-radius:10px; padding:10px 15px; margin:5px 0;
    border-left:4px solid #dc3545; color:#f0b8bc !important; font-size:.88rem;
}
.complaint-card b { color:#f08890 !important; }
.praise-card {
    background:#152d1e; border-radius:10px; padding:10px 15px; margin:5px 0;
    border-left:4px solid #28a745; color:#a8e6bf !important; font-size:.88rem;
}
.praise-card b { color:#6ee89a !important; }
.metric-card {
    background:#1e2535; border-radius:14px; padding:16px 10px; text-align:center;
}
.metric-card h3 { font-size:1.8rem; margin:0; color:#ff9a6c !important; font-weight:800; }
.metric-card p  { font-size:.75rem; color:#9ba3b5 !important; margin:4px 0 0; font-weight:600; }
.section-title {
    font-size:.8rem; font-weight:800; color:#9ba3b5 !important;
    margin:18px 0 8px; text-transform:uppercase; letter-spacing:1px;
}
.info-box {
    background:#2a2010; border-left:4px solid #ff6b35; border-radius:8px;
    padding:11px 15px; margin:10px 0; font-size:.88rem; color:#e8c99a !important;
}
.info-box b { color:#ffb06a !important; }
hr { border-color:#2a3045 !important; }

/* Yulduz ranglari */
.star-1 { color: #C0392B; }
.star-2 { color: #E67E22; }
.star-3 { color: #F1C40F; }
.star-4 { color: #2ECC71; }
.star-5 { color: #27AE60; }
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="title-box">
    <h1>🍽️ RestoPulse v4.0</h1>
    <p>{t('title_sub')}</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([t("tab_analysis"), t("tab_stats"), t("tab_project")])

# ══════════════════════════════════════
# TAB 1 — TAHLIL
# ══════════════════════════════════════
with tab1:

    if "ta_box" not in st.session_state:
        st.session_state["ta_box"] = ""

    st.markdown(f"**{t('samples_label')}**")
    c1, c2, c3 = st.columns(3)
    
    samples = {
        t("sample_terrible"): t("sample_terrible_txt"),
        t("sample_average"):  t("sample_average_txt"),
        t("sample_excellent"): t("sample_excellent_txt"),
    }
    for col, (label, text) in zip([c1, c2, c3], samples.items()):
        with col:
            if st.button(label, use_container_width=True):
                st.session_state["ta_box"] = text
                st.rerun()

    st.markdown(f"### {t('input_header')}")
    user_input = st.text_area(
        "", placeholder=t("input_placeholder"),
        height=120, key="ta_box"
    )

    st.markdown("---")

    if st.button(t("btn_analyze"), type="primary", use_container_width=True):
        if not user_input.strip():
            st.warning(t("warn_empty"))
        else:
            with st.spinner(t("spinner_analyzing")):
                score, label, conf, clean, proba = predict(user_input, model, vectorizer)
                found_aspects, complaints, praises = detect_aspects(user_input, score)

            color = SENTIMENT_COLORS.get(score, "#888")
            stars_str = stars_display(score)
            ui_label = UI_SENTIMENT_LABELS.get(score, label)

            # ── Ball ko'rsatish ──
            st.markdown(f"""
            <div class="score-box" style="border-color:{color}; background:{color}18;">
                <div class="stars" style="color:{color};">{stars_str}</div>
                <div class="big" style="color:{color};">{score} / 5 {t('star_unit')}</div>
                <div class="label" style="color:{color};">{ui_label.upper()}</div>
                <div class="conf">{t('confidence')}: {conf*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

            # ── Yulduz shkalasi (bar chart) ──
            st.markdown(f"**{t('star_prob')}**")
            if proba is not None:
                try:
                    import plotly.graph_objects as go
                    classes = list(model.classes_)
                    colors_bar = [SENTIMENT_COLORS[c] for c in classes]
                    star_labels = [stars_display(c) + f"  ({c}★)" for c in classes]
                    fig = go.Figure(go.Bar(
                        x=star_labels,
                        y=[p*100 for p in proba],
                        marker_color=colors_bar,
                        text=[f"{p*100:.0f}%" if p > 0.03 else "" for p in proba],
                        textposition="outside",
                        textfont=dict(color="#e8eaf0", size=11)
                    ))
                    fig.update_layout(
                        height=240, margin=dict(t=10,b=10,l=10,r=10),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#e8eaf0"),
                        xaxis=dict(title=t("xaxis_stars"), gridcolor="#2a3045"),
                        yaxis=dict(title=t("yaxis_prob"), gridcolor="#2a3045"),
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    st.progress(score / 5)
                    st.caption(f"{score}★ | {ui_label}")
            else:
                st.progress(score / 5)

            st.markdown("---")

            # ── Aspekt tahlili ──
            st.markdown(f'<div class="section-title">{t("aspects_detected_title")}</div>', unsafe_allow_html=True)
            if found_aspects:
                cols = st.columns(min(len(found_aspects), 3))
                for i, (asp_key, matched) in enumerate(found_aspects.items()):
                    meta = ASPECT_META.get(asp_key, {"icon":"💬","label":asp_key})
                    with cols[i % len(cols)]:
                        st.markdown(
                            f'<div class="aspect-card">{meta["icon"]} <b>{meta["label"]}</b>'
                            f'<br><small>{", ".join(matched[:3])}</small></div>',
                            unsafe_allow_html=True
                        )
            else:
                st.markdown(f'<div class="aspect-card">📝 <b>{t("aspects_none")}</b><br><small>{t("aspects_general")}</small></div>', unsafe_allow_html=True)

            # Aspekt munosabati diagrammasi (Diverging Bar Chart)
            st.markdown(f"**{t('aspect_sentiment_title')}**")
            probs = predict_aspect_probabilities(user_input)
            try:
                import plotly.graph_objects as go
                asp_labels = []
                asp_values = []
                bar_colors = []
                text_labels = []
                
                for asp, prob in probs.items():
                    meta = ASPECT_META.get(asp, {"icon": "💬", "label": asp})
                    label = meta["label"]
                    icon = meta["icon"]
                    
                    is_praise = any(p["aspect"] == label for p in praises)
                    is_complaint = any(c["aspect"] == label for c in complaints)
                    
                    if is_praise:
                        val = prob * 100
                        color = "#27AE60" # Yashil (Ijobiy)
                        txt = f"+{val:.0f}%"
                    elif is_complaint:
                        val = -prob * 100
                        color = "#C0392B" # Qizil (Salbiy)
                        txt = f"-{abs(val):.0f}%"
                    else:
                        val = 0.0
                        color = "#7f8c8d"
                        txt = ""
                        
                    if val != 0.0:
                        asp_labels.append(icon + " " + label)
                        asp_values.append(val)
                        bar_colors.append(color)
                        text_labels.append(txt)
                
                if asp_values:
                    sorted_data = sorted(zip(asp_values, asp_labels, bar_colors, text_labels), key=lambda x: x[0])
                    y_val, x_lbl, colors, txts = zip(*sorted_data)
                    
                    fig_asp = go.Figure(go.Bar(
                        x=y_val,
                        y=x_lbl,
                        orientation='h',
                        marker_color=colors,
                        text=txts,
                        textposition="outside",
                        textfont=dict(color="#e8eaf0", size=10)
                    ))
                    fig_asp.update_layout(
                        height=120 + len(y_val)*30, margin=dict(t=10,b=10,l=10,r=40),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#e8eaf0"),
                        xaxis=dict(
                            title=t("xaxis_sentiment"), 
                            range=[-120, 120], 
                            gridcolor="#2a3045",
                            zeroline=True,
                            zerolinecolor="#95a5a6",
                            zerolinewidth=2
                        ),
                        yaxis=dict(gridcolor="#2a3045"),
                        showlegend=False
                    )
                    st.plotly_chart(fig_asp, use_container_width=True)
                else:
                    st.markdown(f'<div class="aspect-card">📝 <b>{t("aspects_none")}</b></div>', unsafe_allow_html=True)
            except Exception as e:
                st.write(probs)

            st.markdown("---")

            # ── Shikoyat / Maqtov ──
            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown(f'<div class="section-title">{t("complaints_title")}</div>', unsafe_allow_html=True)
                if complaints:
                    for c in complaints:
                        st.markdown(f'<div class="complaint-card">{c["icon"]} <b>{c["aspect"]}</b><br><small>"{c["word"]}"</small></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="complaint-card">{t("no_complaints")}</div>', unsafe_allow_html=True)

            with col_r:
                st.markdown(f'<div class="section-title">{t("praise_title")}</div>', unsafe_allow_html=True)
                if praises:
                    for p in praises:
                        st.markdown(f'<div class="praise-card">{p["icon"]} <b>{p["aspect"]}</b><br><small>"{p["word"]}"</small></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="praise-card">{t("no_praise")}</div>', unsafe_allow_html=True)

            with st.expander(t("preprocessing_title")):
                st.info(user_input)
                st.success(clean if clean.strip() else "(bo'sh)")

    # ── CSV ──
    st.markdown("---")
    st.markdown(f"### {t('csv_section_title')}")
    st.markdown(f'<div class="info-box">{t("info_box_csv_col")}</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(t("csv_uploader_label"), type=["csv"])
    if uploaded:
        df_up = pd.read_csv(uploaded)
        if "Review_Text" not in df_up.columns:
            st.error(t("csv_err_column"))
        else:
            with st.spinner(t("spinner_analyzing")):
                results = df_up["Review_Text"].apply(lambda x: predict(str(x), model, vectorizer))
                df_up["Yulduz"]   = results.apply(lambda r: r[0])
                df_up["Daraja"]   = results.apply(lambda r: r[1])
                df_up["Ishonch"]  = results.apply(lambda r: f"{r[2]*100:.0f}%")
                df_up["Guruh"]    = df_up["Yulduz"].apply(score_to_group)
                df_up["Yulduzlar"] = df_up["Yulduz"].apply(stars_display)
                asp_r = df_up["Review_Text"].apply(detect_aspects)
                df_up["Aspektlar"] = asp_r.apply(
                    lambda r: ", ".join([ASPECT_META.get(k,{"label":k})["label"] for k in r[0].keys()]) if r[0] else t("aspects_general")
                )

            total = len(df_up)
            avg   = df_up["Yulduz"].mean()
            s5 = (df_up["Yulduz"] == 5).sum()
            s4 = (df_up["Yulduz"] == 4).sum()
            s3 = (df_up["Yulduz"] == 3).sum()
            s2 = (df_up["Yulduz"] == 2).sum()
            s1 = (df_up["Yulduz"] == 1).sum()

            c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
            with c1: st.markdown(f'<div class="metric-card"><h3>{total}</h3><p>{t("metric_total")}</p></div>', unsafe_allow_html=True)
            with c2: st.markdown(f'<div class="metric-card"><h3 style="color:#ff9a6c">★ {avg:.1f}</h3><p>{t("metric_average")}</p></div>', unsafe_allow_html=True)
            with c3: st.markdown(f'<div class="metric-card"><h3 style="color:#27AE60">{s5}</h3><p>★★★★★</p></div>', unsafe_allow_html=True)
            with c4: st.markdown(f'<div class="metric-card"><h3 style="color:#2ECC71">{s4}</h3><p>★★★★</p></div>', unsafe_allow_html=True)
            with c5: st.markdown(f'<div class="metric-card"><h3 style="color:#F1C40F">{s3}</h3><p>★★★</p></div>', unsafe_allow_html=True)
            with c6: st.markdown(f'<div class="metric-card"><h3 style="color:#E67E22">{s2}</h3><p>★★</p></div>', unsafe_allow_html=True)
            with c7: st.markdown(f'<div class="metric-card"><h3 style="color:#C0392B">{s1}</h3><p>★</p></div>', unsafe_allow_html=True)

            st.markdown("---")
            st.markdown(f"### 📊 {t('tab_stats')}")

            try:
                import plotly.graph_objects as go

                col_a, col_b = st.columns(2)

                with col_a:
                    st.markdown(f"**{t('chart_star_dist')}**")
                    dist = df_up["Yulduz"].value_counts().sort_index()
                    star_x = [stars_display(i) + f"  ({i}★)" for i in dist.index]
                    fig1 = go.Figure(go.Bar(
                        x=star_x,
                        y=dist.values,
                        marker_color=[SENTIMENT_COLORS.get(i,"#888") for i in dist.index],
                        text=dist.values, textposition="outside",
                        textfont=dict(color="#e8eaf0")
                    ))
                    fig1.update_layout(
                        height=300, margin=dict(t=10,b=10,l=10,r=10),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#e8eaf0"),
                        xaxis=dict(gridcolor="#2a3045"),
                        yaxis=dict(title=t("metric_total"), gridcolor="#2a3045"),
                    )
                    st.plotly_chart(fig1, use_container_width=True)

                with col_b:
                    st.markdown(f"**{t('chart_share_by_level')}**")
                    pie_labels = [
                        f"★★★★★ {UI_SENTIMENT_LABELS[5]}",
                        f"★★★★ {UI_SENTIMENT_LABELS[4]}",
                        f"★★★ {UI_SENTIMENT_LABELS[3]}",
                        f"★★ {UI_SENTIMENT_LABELS[2]}",
                        f"★ {UI_SENTIMENT_LABELS[1]}"
                    ]
                    fig2 = go.Figure(go.Pie(
                        labels=pie_labels,
                        values=[int(s5), int(s4), int(s3), int(s2), int(s1)],
                        hole=0.42,
                        marker_colors=["#27AE60","#2ECC71","#F1C40F","#E67E22","#C0392B"],
                        textinfo="label+percent",
                        textfont=dict(size=11, color="white")
                    ))
                    fig2.update_layout(
                        height=300, showlegend=False, margin=dict(t=10,b=10,l=10,r=10),
                        paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e8eaf0")
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                # Aspekt shikoyat/maqtov
                asp_neg, asp_pos = {}, {}
                for _, row in df_up.iterrows():
                    _, comps, praisez = detect_aspects(str(row["Review_Text"]))
                    for c in comps:
                        asp_neg[c["aspect"]] = asp_neg.get(c["aspect"],0)+1
                    for p in praisez:
                        asp_pos[p["aspect"]] = asp_pos.get(p["aspect"],0)+1

                # ── Diverging chart: aspekt bo'yicha Salbiy <-> Ijobiy ──
                st.markdown(f"**{t('aspect_sentiment_title')}**")
                all_aspects = sorted(set(asp_neg.keys()) | set(asp_pos.keys()))
                if all_aspects:
                    neg_vals = [-asp_neg.get(a, 0) for a in all_aspects]
                    pos_vals = [asp_pos.get(a, 0) for a in all_aspects]
                    totals = [asp_neg.get(a,0) + asp_pos.get(a,0) for a in all_aspects]

                    # Eng ko'p tilga olingan aspekt yuqorida bo'lsin
                    order = sorted(range(len(all_aspects)), key=lambda i: totals[i])
                    all_aspects = [all_aspects[i] for i in order]
                    neg_vals = [neg_vals[i] for i in order]
                    pos_vals = [pos_vals[i] for i in order]

                    fig_div = go.Figure()
                    fig_div.add_trace(go.Bar(
                        x=neg_vals, y=all_aspects, orientation="h", name=t("aspect_negative"),
                        marker_color="#C0392B",
                        text=[str(-v) if v != 0 else "" for v in neg_vals],
                        textposition="outside", textfont=dict(color="#e8eaf0")
                    ))
                    fig_div.add_trace(go.Bar(
                        x=pos_vals, y=all_aspects, orientation="h", name=t("aspect_positive"),
                        marker_color="#27AE60",
                        text=[str(v) if v != 0 else "" for v in pos_vals],
                        textposition="outside", textfont=dict(color="#e8eaf0")
                    ))
                    max_range = max([abs(v) for v in neg_vals + pos_vals] + [1]) * 1.25
                    fig_div.update_layout(
                        height=120 + len(all_aspects) * 45,
                        barmode="overlay",
                        margin=dict(t=10, b=10, l=5, r=40),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#e8eaf0"),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                        xaxis=dict(
                            title=t("xaxis_sentiment"),
                            range=[-max_range, max_range],
                            gridcolor="#2a3045",
                            zeroline=True, zerolinecolor="#95a5a6", zerolinewidth=2
                        ),
                        yaxis=dict(gridcolor="#2a3045"),
                    )
                    st.plotly_chart(fig_div, use_container_width=True)
                else:
                    st.info(t("aspects_none"))

            except ImportError:
                st.bar_chart(df_up["Yulduz"].value_counts().sort_index())

            st.markdown("---")
            st.markdown(f"### {t('table_title')}")
            
            flt_options = [
                t("all_filter"),
                "★★★★★ (5)",
                "★★★★ (4)",
                "★★★ (3)",
                "★★ (2)",
                "★ (1)"
            ]
            flt = st.selectbox(t("filter_label"), flt_options)
            if flt == "★★★★★ (5)":
                df_show = df_up[df_up["Yulduz"] == 5]
            elif flt == "★★★★ (4)":
                df_show = df_up[df_up["Yulduz"] == 4]
            elif flt == "★★★ (3)":
                df_show = df_up[df_up["Yulduz"] == 3]
            elif flt == "★★ (2)":
                df_show = df_up[df_up["Yulduz"] == 2]
            elif flt == "★ (1)":
                df_show = df_up[df_up["Yulduz"] == 1]
            else:
                df_show = df_up

            df_show_display = df_show[["Review_Text", "Yulduzlar", "Daraja", "Ishonch", "Aspektlar"]].copy()
            df_show_display.columns = [
                "Review_Text", 
                t("yulduz_label"), 
                t("daraja_label"), 
                t("ishonch_label"), 
                t("aspektlar_label")
            ]
            st.dataframe(df_show_display, use_container_width=True)
            
            csv_out = df_up.to_csv(index=False).encode("utf-8")
            st.download_button(t("btn_download"), csv_out, t("result_file_name"), "text/csv")

# ══════════════════════════════════════
# TAB 2 — STATISTIKA
# ══════════════════════════════════════
with tab2:
    st.markdown(f"### {t('tab_stats_metrics')}")
    if metrics:
        c1,c2,c3 = st.columns(3)
        with c1: st.markdown(f'<div class="metric-card"><h3>{metrics["accuracy"]*100:.1f}%</h3><p>{t("tab_stats_exact_acc")}</p></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="metric-card"><h3>{metrics["tolerance_1"]*100:.1f}%</h3><p>{t("tab_stats_tolerance")}</p></div>', unsafe_allow_html=True)
        with c3: st.markdown(f'<div class="metric-card"><h3>{metrics["mae"]:.2f} ★</h3><p>{t("tab_stats_mae")}</p></div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(f"### {t('tab_stats_dist')}")
        dist = metrics.get("class_distribution", {})
        if dist:
            try:
                import plotly.graph_objects as go
                scores = [1,2,3,4,5]
                counts = [dist.get(str(s), 0) for s in scores]
                star_labels = [stars_display(s) + f"  ({s}★)" for s in scores]
                fig = go.Figure(go.Bar(
                    x=star_labels,
                    y=counts,
                    marker_color=[SENTIMENT_COLORS[s] for s in scores],
                    text=counts, textposition="outside",
                    textfont=dict(color="#e8eaf0")
                ))
                fig.update_layout(height=320, margin=dict(t=10,b=10,l=10,r=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e8eaf0"),
                    xaxis=dict(gridcolor="#2a3045"),yaxis=dict(gridcolor="#2a3045"))
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.write(dist)

        st.markdown(f"### {t('tab_stats_info')}")
        info = {
            t("tab_stats_col_metric"): [
                t("tab_stats_info_train"), 
                t("tab_stats_info_test"), 
                t("tab_stats_info_vocab"), 
                t("tab_stats_info_model"), 
                t("tab_stats_info_scale")
            ],
            t("tab_stats_col_val"): [
                f"{metrics.get('train_size','-')} {t('tab_stats_info_reviews')}",
                f"{metrics.get('test_size','-')} {t('tab_stats_info_reviews')}",
                f"{metrics.get('vocab_size','-')} {t('tab_stats_info_tokens')}",
                metrics.get("model_type","—"),
                metrics.get("scale","—"),
            ]
        }
        st.table(pd.DataFrame(info))
    else:
        st.info(t("tab_stats_not_trained"))

# ══════════════════════════════════════
# TAB 3 — LOYIHA
# ══════════════════════════════════════
with tab3:
    st.markdown("### 🍽️ RestoPulse v4.0")
    st.markdown(t("tab_project_intro"))
    
    col1,col2 = st.columns(2)
    with col1:
        st.markdown(f"**{t('tab_project_heading_stars')}**")
        for s in range(1, 6):
            st.markdown(f"- {stars_display(s)} `{s}★` — {UI_SENTIMENT_LABELS[s]}")
    with col2:
        st.markdown(f"**{t('tab_project_heading_tech')}**")
        for t_item in ["Python 3.x","Scikit-learn (LR)","TF-IDF Vectorizer","Pandas","Streamlit","Plotly"]:
            st.markdown(f"- {t_item}")
    st.markdown("---")
    st.markdown(t("tab_project_footer"))

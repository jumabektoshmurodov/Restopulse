"""
RestoPulse v4.0 - Model Training
1 yulduzdan 5 yulduzgacha sentiment tahlili
"""

import os
import re
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ─────────────────────────────────────────
# YULDUZ TIZIMI (1-5)
# ─────────────────────────────────────────
SENTIMENT_LABELS = {
    1: "Dahshatli",
    2: "Yomon",
    3: "O'rtacha",
    4: "Yaxshi",
    5: "Mukammal",
}

SENTIMENT_COLORS = {
    1: "#C0392B",
    2: "#E67E22",
    3: "#F1C40F",
    4: "#2ECC71",
    5: "#27AE60",
}

SENTIMENT_EMOJI = {
    1: "⭐",
    2: "⭐⭐",
    3: "⭐⭐⭐",
    4: "⭐⭐⭐⭐",
    5: "⭐⭐⭐⭐⭐",
}

# Yulduz to'liq ko'rsatish (to'liq/bo'sh)
def stars_display(score: int) -> str:
    filled = "★" * score
    empty = "☆" * (5 - score)
    return filled + empty

# ─────────────────────────────────────────
# 1. LUG'ATLAR VA QOIDAGA ASOSLANGAN QISM
# ─────────────────────────────────────────
positive_words = {
    'mazali': 2, 'ajoyib': 3, 'yaxshi': 2, "zo'r": 3, 'mukammal': 3,
    'tez': 2, 'muloyim': 2, 'toza': 2, 'chiroyli': 2, 'sifatli': 2,
    'qulay': 2, 'minnatdor': 2, 'rahmat': 1, 'barakali': 2, 'yoqdi': 3,
    'tavsiya': 2, 'qaytaman': 2, 'mamnun': 2, 'shirin': 2, 'yoqimli': 2, 
    'hamyonbop': 2, 'achinmaysiz': 2, "e'tiborli": 2,
    'shinam': 2, 'tezkor': 2, 'xushmuomala': 3, 'chaqqon': 2, 'odobli': 2,
    'halol': 2, 'super': 3, 'klass': 3, 'bomba': 3
}

negative_words = {
    'bemaza': -3, 'yomon': -3, 'dahshatli': -4, 'sovuq': -2, 'iflos': -3,
    'sekin': -2, "qo'pol": -3, 'achchiq': -2, "sho'r": -2, 'eski': -2,
    'isrof': -3, 'zaharlan': -4, 'jirkanch': -4, 'xafa': -2, 'qimmat': -2,
    'kutish': -2, "noto'g'ri": -2, 'qoniqarsiz': -3, "bema'ni": -3,
    'soch': -4, 'chuvalchang': -4, 'chibin': -3, 'pashsha': -3, 'suvarak': -4,
    'tosh': -3, 'shisha': -4, 'zahar': -4, 'chirigan': -3, 'aynigan': -3,
    'kalamush': -4, 'sichqon': -4,
    'pishmagan': -2, 'ogriq': -2, 'shikoyat': -2, 'etiroz': -2, 'aybdor': -2, 'yoqmadi': -3,
    'kuygan': -2, 'achigan': -3, 'sassiq': -3, 'noqulay': -2, 'aldash': -3,
    'aldashdi': -3, 'kasal': -3, 'norozi': -2, 'shovqin': -2, 'tor': -2,
    'sifatsiz': -3, 'yoqimsiz': -2
}

ASPECT_KEYWORDS = {
    "taom": ["taom", "ovqat", "shashlik", "osh", "lag'mon", "kabob", "non", "salat", "porsiya", "mazali", "lazzatli", "ta'mli", "yangi", "fresh", "shirin", "ta'msiz", "sho'r", "achchiq", "chala", "sovuq", "bayiy", "sifatsiz", "yoqimsiz", "kuygan", "achigan", "go'sht", "lavash", "doner", "burger", "pishirilishi", "shurva", "assorti", "sok", "muzqaymoq", "shirinlik", "pishiriq", "kofe", "choy"],
    "xizmat": ["xizmat", "ofitsiant", "xodim", "buyurtma", "muloyim", "iltifotli", "professional", "tez", "e'tibor", "tezkor", "xushmuomala", "chaqqon", "odobli", "sekin", "kech", "unutdi", "qo'pol", "kutdim", "kuryer", "dostavka", "kassir", "kassa", "administrator"],
    "narx": ["narx", "narxi", "pul", "arzon", "adolatli", "chegirma", "bepul", "tejamkor", "qulay", "qimmat", "arzimaydi", "baland", "chek", "schet", "foiz"],
    "muhit": ["muhit", "atmosfera", "bezak", "dizayn", "interior", "interyer", "romantik", "qulay", "tinch", "shinam", "shovqin", "tor", "noqulay", "eskirgan", "musiqa", "svet", "zal", "konditsioner", "issiq", "sovuq"],
    "tozalik": ["toza", "tozalik", "iflos", "hid", "chang", "tozalanmagan", "soch", "suvarak", "chuvalchang", "chibin", "pashsha", "sichqon", "kalamush", "sassiq", "tualet", "hojatxona", "idishlar", "vilka", "qoshiq", "salfetka"],
    "joylashuv": ["manzil", "joylashuv", "parking", "yaqin", "oson", "uzoq", "topish", "lokatsiya", "karta", "avtoturargoh"]
}

def rule_based_sentiment(text):
    text = str(text).lower()
    # Apostroflarni normallashtirish: ’ ‘ ʻ ʼ ´ ` belgilarni ' ga o'zgartirish
    for char in ["’", "‘", "ʻ", "ʼ", "´", "`"]:
        text = text.replace(char, "'")
    # Tinish belgilarini bo'shliq bilan almashtirish (taom,xizmat -> taom xizmat)
    text = re.sub(r"[^\w\s']", " ", text)
    # Matnni faqat so'zlarga ajratish (apostroflarni saqlab qolgan holda)
    words = re.findall(r"\b\w+(?:'\w+)*\b", text)
    
    score = 0
    i = 0
    while i < len(words):
        word = words[i]
        val = 0
        
        if word in positive_words:
            val = positive_words[word]
        elif word in negative_words:
            val = negative_words[word]
            
        # O'ZBEK TILI INKOR QOIDASI: Agar so'zdan keyin "emas" yoki "yo'q" kelsa
        if i + 1 < len(words) and words[i+1] in ['emas', "yo'q"]:
            val = -val # Ishorani teskarisiga o'zgartiramiz (yomon(-3) -> +3)
            if val > 0: 
                val += 1 # Inkor qilingan yomon narsa aslida yaxshi natija beradi
            i += 1 # "emas" so'zini tekshirib bo'ldik, uni o'tkazib yuboramiz
            
        score += val
        i += 1
        
    # Ballarni 1-5 yulduz tizimiga o'tkazish
    if score >= 4: return 5
    elif score > 0: return 4
    elif score == 0: return 3
    elif score > -3: return 2
    else: return 1

# ... (preprocess_ml, train va hokazo funksiyalar o'z joyida qoladi) ...

def predict(text: str, model, vectorizer):
    """Gibrid bashorat (ML + Rule-based), natija: 1-5 yulduz"""
    clean = preprocess_ml(text)
    if not clean.strip():
        return 3, "O'rtacha", 0.5, clean, None

    vec = vectorizer.transform([clean])
    
    # Asosiy ehtimolliklar (original confidence for threshold checks)
    orig_proba = model.predict_proba(vec)[0]
    classes = list(model.classes_)
    ml_score = int(model.predict(vec)[0])
    orig_conf = float(orig_proba[classes.index(ml_score)])
    
    # Model logits va ehtimolliklarni Temperature Scaling yordamida hisoblash (T=0.45)
    logits = model.decision_function(vec)[0]
    T = 0.45
    exp_logits = np.exp(logits / T)
    proba = exp_logits / np.sum(exp_logits)
    conf = float(proba[classes.index(ml_score)])

    # Gibrid Tizim: Qoidali yondashuv ballini olish
    rule_score = rule_based_sentiment(text)
    final_score = ml_score

    # 1. INKOR GAPLAR MANTIQI (emas / yo'q bo'lsa)
    if "emas" in text.lower() or "yo'q" in text.lower():
        if abs(rule_score - ml_score) >= 2:
            final_score = rule_score
            
    # 2. JIDDIY ZIDDIYATLAR HIMOYaSI (emas / yo'q bo'lmasa ham)
    else:
        # Agar matnda umuman salbiy so'z bo'lmasa (rule_score >= 4), 
        # lekin ML modeli asossiz ravishda yomon baho (1 yoki 2) bersa:
        # FAQAT matn qisqa bo'lsa (12 ta so'zdan kam) yoki ML ishonch darajasi o'ta past bo'lsa:
        word_count = len(text.split())
        if rule_score >= 4 and ml_score <= 2:
            if word_count < 12 or orig_conf < 0.35:
                final_score = rule_score
        # Aksinchasi: matn to'la so'kish bo'lsa-yu, ML model uni yaxshi desa (va qisqa bo'lsa):
        elif rule_score <= 2 and ml_score >= 4:
            if word_count < 12 or orig_conf < 0.35:
                final_score = rule_score
        # Agar model o'z qaroriga juda kam ishonayotgan bo'lsa (< 35% original ehtimollik bo'yicha):
        elif orig_conf < 0.35:
            final_score = rule_score

    # Yakuniy natijani 1-5 chegarasida ushlab turish
    final_score = max(1, min(5, final_score))
    label = SENTIMENT_LABELS.get(final_score, "O'rtacha")
    return final_score, label, conf, clean, proba
# ─────────────────────────────────────────
# 2. MACHINE LEARNING QISMI
# ─────────────────────────────────────────
UZBEK_STOP_WORDS = {
    "va", "bilan", "uchun", "ki", "yoki", "ham", "esa", "lekin",
    "chunki", "deb", "ga", "ni", "da", "dan", "ning", "bu", "u",
    "men", "sen", "biz", "siz", "ular", "o'zi", "o'z", "har",
    "bir", "ikki", "ko'p", "oz", "hali", "endi",
    "keyin", "oldin", "yana", "faqat", "ham", "bor",
    "edi", "ekan", "bo'ldi", "qildi", "dedi", "keldi", "bordi",
    "hech", "nima", "qanday", "qaysi", "qachon", "qayer",
    "ammo", "balki", "garchi", "shunda",
}

def preprocess_ml(text: str) -> str:
    text = str(text).lower()
    # Apostroflarni normallashtirish: ’ ‘ ʻ ʼ ´ ` belgilarni ' ga o'zgartirish
    for char in ["’", "‘", "ʻ", "ʼ", "´", "`"]:
        text = text.replace(char, "'")
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\d+", " ", text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in UZBEK_STOP_WORDS and len(w) > 1]
    return " ".join(tokens)


def score_old_to_star(old_score: int) -> int:

    if old_score <= -4:
        return 1
    elif old_score <= -2:
        return 2
    elif old_score <= 0:
        return 3
    elif old_score <= 2:
        return 4
    else:
        return 5


def train(csv_path: str = "data/reviews.csv", model_dir: str = "model"):
    os.makedirs(model_dir, exist_ok=True)

    print("=" * 55)
    print("RestoPulse v4.0 — Model O'qitish (1-5 yulduz)")
    print("=" * 55)

    # 1. Data yuklash
    print("\n1. Ma'lumotlar yuklanmoqda...")
    df = pd.read_csv(csv_path)

    df["Sentiment"] = pd.to_numeric(df["Sentiment"], errors="coerce")
    df = df.dropna(subset=["Sentiment"])
    df["Sentiment"] = df["Sentiment"].astype(int)

    # Agar dataset -5..+5 da bo'lsa, yulduzga o'tkazamiz
    if df["Sentiment"].min() < 1 or df["Sentiment"].max() > 5:
        print("   Eski -5..+5 tizimdan 1-5 yulduzga konvertatsiya qilinmoqda...")
        df["Sentiment"] = df["Sentiment"].apply(score_old_to_star)

    print(f"   Jami: {len(df)} ta sharh")
    print("\n   Yulduz bo'yicha:")
    for s in range(1, 6):
        cnt = (df["Sentiment"] == s).sum()
        bar = "=" * (cnt // 10 if cnt >= 10 else cnt)
        stars_ascii = "*" * s + " " * (5 - s)
        print(f"   [{stars_ascii}] ({s} yulduz): {cnt:4d}  {bar}")

    # 2. Preprocessing
    print("\n2. Preprocessing...")
    df["Clean"] = df["Review_Text"].apply(preprocess_ml)

    # 3. TF-IDF
    print("\n3. TF-IDF vektorlashtirish...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=8000,
        min_df=1,
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(df["Clean"])
    y = df["Sentiment"].values
    print(f"   Lug'at: {len(vectorizer.vocabulary_)} token")
    print(f"   Matritsa: {X.shape}")

    # 4. Train/Test split
    print("\n4. 80/20 ajratish...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # 5. Model
    print("\n5. Logistic Regression o'qitilmoqda...")
    model = LogisticRegression(
        C=1.5,
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    )
    model.fit(X_train, y_train)
    print("   [OK] O'qitildi")

    # 6. Baholash
    print("\n6. Natijalar:")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    tolerance_1 = np.mean(np.abs(y_pred - y_test) <= 1)
    mae = np.mean(np.abs(y_pred - y_test))

    print(f"\n   Exact Accuracy     : {acc*100:.1f}%")
    print(f"   +/- 1 yulduz aniqlik  : {tolerance_1*100:.1f}%")
    print(f"   O'rtacha xato (MAE): {mae:.2f} yulduz")

    # 7. Saqlash
    print("\n7. Saqlash...")
    with open(f"{model_dir}/model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(f"{model_dir}/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print("\n8. Aspekt modellarini o'qitish va saqlash...")
    for aspect, keywords in ASPECT_KEYWORDS.items():
        y_aspect = df["Clean"].apply(
            lambda text: 1 if any(bool(re.search(rf"(?<![a-zA-Z']){re.escape(kw)}(?![a-zA-Z'])", text)) for kw in keywords) else 0
        ).values
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf.fit(X, y_aspect)
        with open(f"{model_dir}/model_{aspect}.pkl", "wb") as f:
            pickle.dump(clf, f)
        print(f"   [OK] {model_dir}/model_{aspect}.pkl")

    metrics = {
        "accuracy": round(acc, 4),
        "tolerance_1": round(tolerance_1, 4),
        "mae": round(float(mae), 4),
        "train_size": int(X_train.shape[0]),
        "test_size": int(X_test.shape[0]),
        "vocab_size": len(vectorizer.vocabulary_),
        "model_type": "Logistic Regression (multinomial)",
        "scale": "1 to 5 stars",
        "class_distribution": {
            str(s): int((df["Sentiment"] == s).sum()) for s in range(1, 6)
        }
    }
    with open(f"{model_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"   [OK] {model_dir}/model.pkl")
    print(f"   [OK] {model_dir}/vectorizer.pkl")
    print(f"   [OK] {model_dir}/metrics.json")
    print("\nTAYYOR! -> streamlit run app.py")
    return metrics


def load_model(model_dir: str = "model"):
    with open(f"{model_dir}/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(f"{model_dir}/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer
if __name__ == "__main__":
    if not os.path.exists("data/reviews.csv"):
        print("Dataset yo'q. Yaratilmoqda...")
        try:
            from dataset import build_dataset
            build_dataset()
        except ImportError:
            print("XATO: 'dataset.py' topilmadi yoki 'build_dataset' funksiyasi yo'q.")
            exit(1)
    train()

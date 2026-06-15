# 🍽️ RestoPulse — AI-Powered Uzbek Restaurant Sentiment & Aspect Analyzer
*(Bilingual Documentation / Ikki tilli hujjatlar)*

An AI system that classifies Uzbek restaurant reviews into a 1-to-5 star rating scale and extracts feedback across 6 operational aspects: Food, Service, Price, Atmosphere, Cleanliness, and Location. 
Created as part of the BTEC Level 3 IT — Unit 21: Introduction to Artificial Intelligence course.

O'zbek tilidagi restoran sharhlarini 1 dan 5 yulduzgacha baholaydigan va 6 ta soha (Taom, Xizmat, Narx, Muhit, Tozalik, Joylashuv) bo'yicha tahlil qiladigan AI tizimi.
BTEC Level 3 IT — Unit 21: Introduction to Artificial Intelligence loyihasi.

---

## 🌍 Language Options / Til sozlamalari
The application now supports **English** and **Uzbek**! You can toggle the language from the sidebar within the running web application.

Ilova endi **Ingliz** va **O'zbek** tillarini qo'llab-quvvatlaydi! Tilni sidebar (yon panel) orqali o'zgartirishingiz mumkin.

---

## 📁 File Structure / Fayl tuzilishi

```
restopulse/
├── dataset.py       # 5000+ synthetic review generator / 5000+ sharh generatsiya qilish
├── train_model.py   # Preprocessing & Model training / Preprocessing + Model o'qitish
├── app.py           # Streamlit Web UI / Streamlit veb-interfeys
├── Data_scraper.py  # Google Maps Selenium Scraper / Google Maps sharh yig'uvchi
├── requirements.txt # Dependencies / Kutubxonalar
├── data/
│   └── reviews.csv  # Dataset (built automatically / dataset.py tomonidan yaratiladi)
└── model/
    ├── model.pkl       # Trained Logistic Regression model / O'qitilgan model
    ├── vectorizer.pkl  # TF-IDF Vectorizer
    └── metrics.json    # Model evaluation metrics / Model natijalari
```

---

## 🚀 Running the Project / Ishga tushirish

### 1. Install Dependencies / Kutubxonalarni o'rnatish
```bash
pip install -r requirements.txt
```

### 2. Generate Dataset (Optional) / Dataset yaratish (Ixtiyoriy)
```bash
python dataset.py
```
*(This is run automatically by the training script if reviews.csv does not exist / reviews.csv mavjud bo'lmasa, train_model.py buni avtomatik chaqiradi)*

### 3. Train Model / Modelni o'qitish
```bash
python train_model.py
```

### 4. Run Streamlit UI / Streamlitni ishga tushirish
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser. / Brauzerda `http://localhost:8501` manzilini oching.

---

## 🔬 Model Specifications / Model haqida

| Metric / Ko'rsatkich | Description / Tavsif |
| :--- | :--- |
| **Algorithm / Algoritm** | Logistic Regression (Multinomial) |
| **Vectorization / Vektorlashtirish** | TF-IDF Vectorizer (1-3 n-grams, max 8000) |
| **Train/Test Split** | 80% / 20% |
| **Target Accuracy / Maqsadli aniqlik** | ≥ 95% (Exact), ≥ 98% (±1 star / yulduz) |
| **Scale / Shkala** | 1 to 5 Stars (5 classes) / 1 dan 5 yulduzgacha (5 ta sinf) |
| **Preprocessing** | Lowercase → Apostrophe Norm → StopWords → TF-IDF |

---

## 📊 Preprocessing Steps / Preprocessing bosqichlari

1. **Lowercasing** — Converts all letters to lowercase / barcha harflarni kichik harfga o'tkazish.
2. **Apostrophe Normalization** — Converts variations (`’`, `‘`, `ʻ`, `ʼ`, `` ` ``) into standard `'` / apostroflarni standart `'` ga o'tkazish.
3. **Stop Words** — Filters out noise words (e.g., "va", "bilan", "uchun") / shovqin so'zlarni filtrlash.
4. **TF-IDF Vectorizer** — Converts text into numerical TF-IDF feature matrices / matnni raqamli TF-IDF matritsasiga o'tkazish.

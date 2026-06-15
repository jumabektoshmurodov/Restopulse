"""
RestoPulse v5.0 - Realistic Dataset Generator
Muvaffaqiyatli o'qitish va overfitting (99.9% -> 80-85% gacha) muammosini hal qilish uchun.
"""

import pandas as pd
import random
import os

# ─────────────────────────────────────────
# SEED SHARHLAR (HAQIQIY FOYDALANUVChILAR NUSXASI)
# ─────────────────────────────────────────
REVIEWS = {
    1: [
        "Hayotimda bunday dahshatli restoran ko'rmaganman, hammasi falokat edi.",
        "Ovqatdan zaharlandim, kasalxonaga tushishimga sal qoldi.",
        "Juda iflos, idishlar yuvilmagan, dahshatli holat.",
        "Xodimlar haqoratomuz gapirdi, menejer ham yomon munosabat ko'rsatdi.",
        "Taomda chuvalchang chiqdi, buni hech qachon kechira olmayman.",
        "Pul qaytarib berilmadi, xizmat ko'rsatish nol darajasida.",
        "Ovqat ayniydigan darajada eski edi, ishtaham butunlay ketib qoldi.",
        "Bu restoran yopilishi kerak, sanitariya qoidalari buzuq.",
        "Dostavka judayam kech keldi, ovqatlar muzlab qolibdi. Xizmat ko'rsatish 0.",
        "Stolda suvarak yuribdi! Ofitsiantga aytsam kuladi, bu qanaqa joy o'zi?!",
        "Zaxarlanishga sal qoldi, oshqozonim og'riyapti. Go'shti chirigan ekan.",
        "Ovqatdan soch chiqdi, daxshat iflos joy. Hech kimga tavsiya qilmayman."
    ],
    2: [
        "Yomon tajriba, yana kelmasam kerak.",
        "Xizmat juda sekin va xodimlar umuman e'tiborsiz edi.",
        "Taom ta'msiz va sovuq edi, kutganimga arzimadi.",
        "Porsiya juda kichik, bergan pulimga achindim.",
        "Buyurtmani unutib qo'yishdi, qayta eslatishga to'g'ri keldi.",
        "Shovqin juda kuchli edi, sherigim bilan gaplasha olmadik.",
        "Narxi ko'rsatilgandan ko'p chiqdi, hech kim tushuntirmadi.",
        "Taom achchiq edi, oldindan aytganimga qaramay e'tibor berishmadi.",
        "Qimmat ekan, lekin xizmat sifati judayam past.",
        "Nonlar qattiq va eskirgan edi, yangi non olib kelishmadi.",
        "Konditsioner ishlamasdan juda issiq bo'lib ketdi ichkari, o'tirib bo'lmadi."
    ],
    3: [
        "Biroz mamnun emasman, ba'zi narsalar yaxshilanishi kerak.",
        "Asosan yaxshi, lekin xizmat biroz sekinroq edi.",
        "Taom yomon emas, lekin narxi biroz qimmat tuyuldi.",
        "O'rtacha darajada, hech narsa alohida ajralib turmaydi.",
        "Narxi maqbul, lekin sifat biroz pastroq.",
        "Tushlik uchun kelgandim, qorin to'ydi, lekin katta taassurot ololmadim.",
        "Joylashuvi qulay, lekin muhit biroz shovqinli.",
        "Restoran oddiy, na yaxshi, na yomon.",
        "Normalniy joy, obedda tez-tez kelib turamiz.",
        "Taom unchalik mazali emas, lekin xizmat muloyimligi uchun 3 yulduz.",
        "Narxlari hamyonbop, taomlari o'rtacha darajada."
    ],
    4: [
        "Umuman olganda yaxshi edi, ijobiy taassurot qoldirdi.",
        "Taom yaxshi pishirilgan, yana kelish mumkin.",
        "Xizmat odatdagidan yaxshiroq va tez bo'ldi.",
        "Narxi biroz qimmat, lekin sifatiga to'liq arziydi.",
        "Muhit qulay, oilaviy o'tirish uchun yaxshi joy.",
        "Xodimlar muloyim va e'tiborli ekan, yoqdi.",
        "Porsiyalar katta va to'yimli, narxi ham adolatli.",
        "Toza va qulay atmosfera, mamnun bo'ldim.",
        "Pizza juda zo'r chiqibdi! Porsiyasi ham katta va to'ydiradigan ekan.",
        "Shinamgina joy, bolalar o'yin maydonchasi borligi qulay bo'ldi."
    ],
    5: [
        "Juda yaxshi restoran, har safar bu yerdan mamnun ketaman.",
        "Taomlar nihoyatda mazali, xizmat eng yuqori professional darajada.",
        "Muhit chiroyli va qulay, ovqat shunchaki ajoyib.",
        "Narxi adolatli, sifat yuqori, barchaga tavsiya qilaman.",
        "Xodimlar juda muloyim va tezkor, xizmat ko'rsatish zo'r.",
        "Oilaviy kecha uchun ideal joy, hammamizga juda yoqdi.",
        "Hayotimda yegan eng shirin taomlardan biri bo'ldi.",
        "Besh yulduz ham kam bu joyga, mukammal!",
        "Kabob judayam zo'r chiqibdi, go'shti og'izda erib ketadi. Rahmat povar akaga!",
        "Hamma narsa ideal darajada, eng sevimli restoranlarimizdan biri."
    ],
}

# ─────────────────────────────────────────
# LUG'ATLAR (FILLS)
# ─────────────────────────────────────────
FILLS = {
    # Salbiy
    "neg_issue": ["taomlari aynigan", "xizmati dahshatli", "gigiyena umuman yo'q", "xodimlari juda qo'pol", "atrof iflos", "go'shti chirigan ekan"],
    "neg_consequence": ["kasal bo'lib qoldim", "pulum isrof bo'ldi", "qimmatli vaqtim ketdi", "kayfiyatim butunlay buzildi", "bayramim barbod bo'ldi", "zaharlandim"],
    "neg_problem": ["Taom muzdek sovuq keldi", "Buyurtmamizni unutishdi", "Stolimiz yig'ishtirilmagan edi", "Buyurtmani bir soat kutdik"],
    "neg_feeling": ["juda xafa bo'ldim", "hafsalam pir bo'ldi", "g'azabim keldi", "qattiq asabiylashdim"],
    "neg_detail": ["sifati juda past edi", "narxi haddan tashqari qimmat", "ofitsiantlar umuman qaramadi", "muhit juda noqulay va shovqinli"],
    "neg_food_serv": ["ta'msiz edi", "sovuq keldi", "sho'r bo'lib ketgan", "achchiq edi", "xom ekan", "kuyib ketgan ekan", "eski edi"],
    "neg_service": ["sekin", "qo'pol", "e'tiborsiz", "nolga teng", "juda yomon"],
    "neg_minor": ["kutish uzoqroq bo'ldi", "stol sal iflos ko'rindi", "musiqa baland chalindi", "porsiyalar juda kichik edi"],
    "contamination": ["soch", "suvarak", "chibin", "pashsha", "tosh", "shisha", "chuvalchang"],

    # Neytral
    "neu_obs": ["alohida tavsiya qilmayman", "odatiy joy", "eslab qolarli hech narsa yo'q", "faqat qorin to'yg'azish uchun borsa bo'ladi"],
    "neu_comment": ["shoshilganlar uchun qulay", "yo'l usti uchun normal", "narxiga yarasha xizmat", "ko'p narsa kutib bormaslik kerak"],

    # Ijobiy
    "pos_detail": ["xizmat tez edi", "narxi adolatli va hamyonbop", "joy juda toza va shinam ekan", "porsiyalar kutilganidan katta edi", "xodimlar chaqqon ishladi"],
    "pos_food_serv": ["nihoyatda mazali edi", "og'izda eriydi", "yangi va issiq keldi", "pishirilishi a'lo darajada", "judayam maza qildik"],
    "pos_reaction": ["albatta yana kelaman", "barchaga tavsiya qilaman", "do'stlarimni ham olib kelaman", "ko'p marta kelishga arziydi", "kuningizni ajoyib o'tkazasiz"],

    # Aspektlar
    "aspect": ["Taom", "Xizmat", "Muhit", "Narx", "Tozalik", "Joylashuv", "Dizayn"],
    "aspect2": ["atmosferasi", "xodimlari", "porsiyalari", "tezligi", "narx-navosi", "interyeri"]
}

# ─────────────────────────────────────────
# REALTISTIK TEMPLATES (KATTAROQ XILMA-XILLIK)
# ─────────────────────────────────────────
TEMPLATES = {
    1: [
        "Bu restoran {neg_issue}, {neg_consequence}. Hech kimga tavsiya qilmayman.",
        "{neg_problem} bo'ldi, {neg_feeling}. Bu shunchaki dahshat.",
        "Umrimda ko'rgan eng yomon joy, {neg_detail}.",
        "Asabim buzildi! {aspect} {neg_food_serv}, {neg_consequence}.",
        "Ovqatdan {contamination} chiqdi! Gigiyena nol, daxshat yomon joy.",
        "Idishlar iflos, {aspect} dahshatli darajada yomon, {neg_consequence}."
    ],
    2: [
        "Umuman qoniqarsiz tajriba, {neg_detail}.",
        "Taom {neg_food_serv}, xizmat esa {neg_service}. Pulum isrof bo'ldi.",
        "Yaxshi emas, {neg_minor}.",
        "Kutganimdek chiqmadi, {neg_problem}.",
        "Gigiyenasi yaxshi emas, taomdan {contamination} chiqdi. Narxi esa qimmat.",
        "Ofitsiantlar juda sekin, {aspect} ham sovuq keldi, mamnun emasman."
    ],
    3: [
        "Oddiy restoran, {neu_obs}.",
        "Na yaxshi na yomon, {neu_comment}.",
        "O'rtacha daraja, {aspect} normal edi, lekin {neg_minor}.",
        "Tushlik qilib ketishga bo'ladi, {neu_comment}.",
        "{aspect} {pos_food_serv} edi, lekin xizmati {neg_service}.",
        "Narxi arzon emas, lekin ovqatlari normalniy, {neu_comment}."
    ],
    4: [
        "Umuman yaxshi, {pos_detail}.",
        "{aspect} yoqdi, {pos_reaction}.",
        "Yaxshi restoran ekan, {aspect} {pos_food_serv} edi.",
        "Kutganimdan yaxshiroq, {pos_detail}. {pos_reaction}.",
        "Xizmat tezligi yaxshi, {aspect2} ham qulay. Tavsiya qilaman.",
        "Ovqatlari mazali, {pos_detail}. Narxi biroz qimmatroq."
    ],
    5: [
        "Mukammal! {pos_detail}. Hech narsa yoqmagan joyi yo'q.",
        "Eng yaxshi restoranlardan biri, {aspect} va {aspect2} ajoyib edi.",
        "Hayotimda unutilmas tajriba, {pos_food_serv}. {pos_reaction}.",
        "Barchasi zo'r! {aspect} mukammal, {pos_detail}.",
        "Kaboblariga gap yo'q, {pos_food_serv}. Xodimlari ham juda e'tiborli.",
        "Taomlari juda yangi va mazali, {pos_reaction}. Rahmat povar akalarga!"
    ]
}

# ─────────────────────────────────────────
# OG'ZAKI SHOVQIN VA IMLE XATOLIK KIRITISH (NOISE INJECTOR)
# ─────────────────────────────────────────
def inject_typo_and_slang(text: str) -> str:
    """Haqiqiy foydalanuvchilar nutqini simulyatsiya qilish uchun shovqin kiritish."""
    # 1. Unicode apostroflarini normallashtirish va ba'zan tasodifiy o'chirish (30% ehtimollik)
    if random.random() < 0.3:
        text = text.replace("'", "").replace("’", "").replace("‘", "").replace("`", "")
        
    # 2. Sleng va og'zaki so'zlarga almashtirish (Tashkent/Internet slang)
    slang_map = {
        "restoran": ["kafe", "oshxona", "joy", "zavedeniye"],
        "ofitsiant": ["xodim", "ofitsiantlar", "ofitsant", "povar"],
        "taom": ["ovqat", "blyuda", "eda"],
        "zo'r": ["zor", "bomba", "klass", "daxshat", "super", "zoorr"],
        "yomon": ["yomon", "ploxoy", "nol", "daxshat"],
        "rahmat": ["raxmat", "spasibo"],
        "bilan": ["bln"],
        "emas": ["emas", "mas"]
    }
    
    words = text.split()
    for i, w in enumerate(words):
        w_clean = w.lower().replace("'", "").replace("’", "").replace("‘", "")
        if w_clean in slang_map and random.random() < 0.25:
            words[i] = random.choice(slang_map[w_clean])
            
    # 3. Tasodifiy harf xatoliklari (h va x o'rnini almashtirish)
    text_res = " ".join(words)
    if random.random() < 0.15:
        # O'zbek tilida 'x' va 'h' ni adashtirish juda ko'p uchraydi
        text_res = text_res.replace("x", "h").replace("h", "x")
        
    return text_res


def generate_reviews(label: int, count: int) -> list:
    """Belgilangan yulduz (label) uchun shablonlardan sharhlar generatsiya qiladi."""
    reviews = []
    templates = TEMPLATES.get(label, ["{aspect} yaxshi emas."])
    
    for _ in range(count):
        tmpl = random.choice(templates)
        result = tmpl
        for key, options in FILLS.items():
            ph = "{" + key + "}"
            # Shablon ichida bir xil placeholder ko'p bo'lsa, hammasini almashtirish
            while ph in result:
                result = result.replace(ph, random.choice(options), 1)
        
        # Realistik shovqin (sleng va imlo xatolarini) kiritamiz
        result = inject_typo_and_slang(result)
        reviews.append({"Review_Text": result.strip(), "Sentiment": label})
        
    return reviews


def build_dataset(output_path: str = "data/reviews.csv"):
    random.seed(42)
    os.makedirs("data", exist_ok=True)
    data = []

    # 1. Qo'lda yozilgan real sharhlarni qo'shish
    for label, reviews in REVIEWS.items():
        for text in reviews:
            data.append({"Review_Text": text, "Sentiment": label})

    # 2. Har bir daraja uchun 1000 tadan generatsiya qilish (Jami 5000+ ta)
    TARGET_PER_CLASS = 1000
    for label in range(1, 6):
        data += generate_reviews(label, TARGET_PER_CLASS)

    # Ma'lumotlarni tasodifiy aralashtirish
    random.shuffle(data)
    
    # 3. Label noise simulyatsiyasi (tasodifiy yulduzni +/-1 oralig'ida o'zgartirish: 15% ehtimollik)
    # Bu insonlar baho berishda bir xil sharhga har xil yulduz (masalan 4 yoki 5) qo'yishini simulyatsiya qiladi.
    for row in data:
        if random.random() < 0.15:
            noise = random.choice([-1, 1])
            new_label = row["Sentiment"] + noise
            if 1 <= new_label <= 5:
                row["Sentiment"] = new_label
                
    # CSV faylga saqlash
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False, encoding='utf-8')

    print("=" * 50)
    print(f"[OK] DATASET MUVAFFAQIYATLI YARATILDI!")
    print(f"Fayl manzili: {output_path}")
    print(f"Jami sharhlar soni: {len(df)} ta")
    print("\nYulduzlar bo'yicha taqsimot:")
    
    labels_text = {
        1: "Dahshatli", 
        2: "Yomon",
        3: "O'rtacha", 
        4: "Yaxshi",
        5: "Mukammal",
    }
    
    for s in range(1, 6):
        cnt = (df["Sentiment"] == s).sum()
        print(f"  {s} yulduz ({labels_text[s]:10s}): {cnt} ta")
    print("=" * 50)
    
    return df

if __name__ == "__main__":
    build_dataset()
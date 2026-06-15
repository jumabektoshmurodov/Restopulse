"""
RestoPulse — Google Maps Selenium Scraper
Bepul, API kalitsiz O'zbekcha/Ruscha restoran sharhlari yig'ish
pip install selenium webdriver-manager pandas langdetect
"""

import time
import csv
import os
import random
from datetime import datetime

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException


# ─────────────────────────────────────────
# SOZLAMALAR
# ─────────────────────────────────────────

SEARCH_QUERIES = [
    "Besh Qozon Pilaf Center"
]

MAX_PLACES_PER_QUERY  = 5  # Har qidiruv uchun joylar soni
MAX_REVIEWS_PER_PLACE = 50  # Har joydan sharhlar soni
OUTPUT_FILE = "test_reviews.csv"
HEADLESS    = False           # False = brauzer ko'rinadi (tavsiya)
SKIP_ALREADY_SCRAPED = True   # True = oldin olingan joylarni o'tkazib yuborish, False = o'tkazib yubormaslik

# Qaysi tillar qabul qilinsin (langdetect kodlari)
# "uz" = o'zbek, "ru" = rus, ikkalasi ham kerak bo'lsa ikkisini yozing
ALLOWED_LANGS = {}

# ─────────────────────────────────────────
# YORDAMCHI
# ─────────────────────────────────────────
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def human_sleep(a=1.0, b=2.5):
    time.sleep(random.uniform(a, b))

def rating_to_star(text: str) -> int:
    import re
    nums = re.findall(r"[\d]+[.,]?[\d]*", text.replace(",", "."))
    if nums:
        return max(1, min(5, round(float(nums[0]))))
    return 3

def detect_lang(text: str) -> str:
    """Matn tilini aniqlash. Xato bo'lsa 'unknown' qaytaradi."""
    try:
        from langdetect import detect
        return detect(text)
    except Exception:
        return "unknown"

def is_allowed_lang(text: str) -> bool:
    """Faqat ruxsat etilgan tillardagi sharhlarni qabul qilish"""
    if not ALLOWED_LANGS:
        return True   # Bo'sh ro'yxat = barchasi qabul
    lang = detect_lang(text)
    return lang in ALLOWED_LANGS

# ─────────────────────────────────────────
# BRAUZER — O'zbek tili majburlangan
# ─────────────────────────────────────────
def make_driver(headless=HEADLESS):
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)

    # Til va hudud: O'zbek tili, O'zbekiston
    opts.add_argument("--lang=uz-UZ")
    opts.add_experimental_option("prefs", {
        "intl.accept_languages": "uz-UZ,uz,ru-RU,ru",
        "profile.default_content_setting_values.geolocation": 2,
    })
    opts.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
    driver  = webdriver.Chrome(options=opts)

    driver.execute_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )
    return driver

# ─────────────────────────────────────────
# JOYLARNI QIDIRISH
# ─────────────────────────────────────────
def search_places(driver, query: str) -> list:
    """
    Google Maps da qidirib, joy URL larini qaytaradi.
    hl=uz  → interfeys O'zbekcha
    gl=UZ  → O'zbekiston hududi (mahalliy natijalar)
    """
    url = (
        f"https://www.google.com/maps/search/"
        f"{query.replace(' ', '+')}/"
        f"?hl=uz&gl=UZ"
    )
    driver.get(url)
    human_sleep(2, 4)

    # Cookie/consent popup yopish
    for btn_text in ["Rad etish", "Qabul qilish", "Reject all", "Accept all"]:
        try:
            btn = driver.find_element(By.XPATH, f"//button[contains(.,'{btn_text}')]")
            btn.click()
            human_sleep(1, 2)
            break
        except NoSuchElementException:
            pass

    # Agar bitta restoranga to'g'ridan-to'g'ri yo'naltirgan bo'lsa
    current_url = driver.current_url
    if "/maps/place/" in current_url:
        log("   [Direct Redirect] Qidiruv bitta aniq joyga yo'naltirildi.")
        return [current_url]

    place_links = []
    scroll_attempts = 0
    max_scrolls = MAX_PLACES_PER_QUERY // 4 + 4

    while len(place_links) < MAX_PLACES_PER_QUERY and scroll_attempts < max_scrolls:
        cards = driver.find_elements(By.CSS_SELECTOR, "a[href*='/maps/place/']")
        for card in cards:
            href = card.get_attribute("href")
            if href and href not in place_links:
                place_links.append(href)

        if len(place_links) >= MAX_PLACES_PER_QUERY:
            break

        try:
            panel = driver.find_element(By.CSS_SELECTOR, "div[role='feed']")
            driver.execute_script("arguments[0].scrollTop += 1200", panel)
        except NoSuchElementException:
            driver.execute_script("window.scrollBy(0, 1200)")

        human_sleep(1.5, 2.5)
        scroll_attempts += 1

        try:
            driver.find_element(
                By.XPATH, "//*[contains(text(),'end of list') or contains(text(),\"Ro'yxat tugadi\")]"
            )
            break
        except NoSuchElementException:
            pass

    return list(dict.fromkeys(place_links))[:MAX_PLACES_PER_QUERY]

# ─────────────────────────────────────────
# SHARHLARNI YIG'ISH
# ─────────────────────────────────────────
def scrape_reviews(driver, place_url: str, seen_places: set = None) -> list:
    """
    Bitta joy sahifasini ochib, sharhlarni yig'adi.
    URL ga ?hl=uz&gl=UZ qo'shiladi — Google o'zbek tilini ko'rsatadi.
    """
    # URL ga til parametrlarini qo'shish
    sep = "&" if "?" in place_url else "?"
    url = place_url + sep + "hl=uz&gl=UZ"
    driver.get(url)
    human_sleep(2, 3)

    # Joy nomi
    place_name = "Noma'lum"
    try:
        place_name = WebDriverWait(driver, 6).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "h1.DUwDvf, h1"))
        ).text.strip()
    except TimeoutException:
        pass

    if seen_places and place_name in seen_places:
        return [{"Place_Name": place_name, "skip": True}]

    # "Sharhlar" tabiga o'tish
    # O'zbekcha, Ruscha va Inglizcha interfeyslar uchun universal XPath
    try:
        reviews_tab = WebDriverWait(driver, 8).until(
            EC.element_to_be_clickable((
                By.XPATH,
                "//button[contains(@aria-label,'Sharh') or contains(@aria-label,'Отзыв') or contains(@aria-label,'Review')] | "
                "//div[@role='tab'][contains(.,'Sharh') or contains(.,'Отзыв') or contains(.,'Review')] | "
                "//*[text()='Sharhlar' or text()='Отзывы' or text()='Reviews']"
            ))
        )
        reviews_tab.click()
        human_sleep(2, 3)
    except Exception as e:
        log(f"     [WARNING] Sharhlar tabiga o'tib bo'lmadi: {e}")

    # Eng so'nggi sharhlar (yangi sharhlar ko'proq mahalliy bo'ladi)
    try:
        sort_btn = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((
                By.XPATH,
                "//button[contains(@aria-label,'saralash') or contains(@aria-label,'Сортировать') or contains(@aria-label,'Sort')] | "
                "//button[@data-value='Sort'] | "
                "//*[text()='Saralash' or text()='Сортировать' or text()='Sort']"
            ))
        )
        sort_btn.click()
        human_sleep(1, 2)
        # "Eng yangi" ni tanlash
        newest = driver.find_element(
            By.XPATH,
            "//li[contains(.,'Eng yangi') or contains(.,'Новые') or contains(.,'Newest')] | "
            "//div[@role='menuitem'][contains(.,'Eng yangi') or contains(.,'Новые') or contains(.,'Newest')]"
        )
        newest.click()
        human_sleep(1.5, 2.5)
    except Exception as e:
        log(f"     [WARNING] Saralash tugmasini bosib bo'lmadi: {e}")

    # Sharhlarni yig'ish
    reviews_data = []
    last_count   = 0
    no_new_count = 0
    skipped_lang = 0

    while len(reviews_data) < MAX_REVIEWS_PER_PLACE:
        # "Ko'proq ko'rish" tugmalarini bosish
        more_btns = driver.find_elements(
            By.CSS_SELECTOR,
            "button.w8nwRe, button[jsaction*='pane.review.expandReview']"
        )
        for btn in more_btns:
            try:
                driver.execute_script("arguments[0].click()", btn)
            except Exception:
                pass

        review_els = driver.find_elements(
            By.CSS_SELECTOR, "div.jftiEf, div[data-review-id]"
        )

        for el in review_els:
            if len(reviews_data) >= MAX_REVIEWS_PER_PLACE:
                break
            try:
                # Matnni olish
                try:
                    text = el.find_element(
                        By.CSS_SELECTOR,
                        "span.wiI7pd, span[class*='review-full-text']"
                    ).text.strip()
                except NoSuchElementException:
                    spans = el.find_elements(By.CSS_SELECTOR, "span")
                    text = max((s.text.strip() for s in spans), key=len, default="")

                if not text or len(text) < 15:
                    continue

                # ── TIL FILTRI ──
                if ALLOWED_LANGS and not is_allowed_lang(text):
                    skipped_lang += 1
                    continue

                # Reytingni olish
                star_val = 3
                try:
                    star_el = el.find_element(
                        By.CSS_SELECTOR, "span[aria-label*='yulduz'], span[aria-label*='star'], span[role='img']"
                    )
                    aria = star_el.get_attribute("aria-label") or ""
                    star_val = rating_to_star(aria)
                except NoSuchElementException:
                    pass

                entry = {
                    "Review_Text":   text,
                    "Sentiment":     star_val,
                    "Place_Name":    place_name,
                    "Google_Rating": star_val,
                }

                if not any(r["Review_Text"] == text for r in reviews_data):
                    reviews_data.append(entry)

            except Exception:
                continue

        # Pastga scroll qilish uchun mos panelni topish (universal)
        try:
            panel = driver.find_element(
                By.XPATH,
                "//div[@role='feed'] | "
                "//div[contains(@class, 'm6QErb') and @tabindex='-1'] | "
                "//div[contains(@class, 'DxyBCb')]"
            )
            driver.execute_script("arguments[0].scrollTop += 2000", panel)
        except NoSuchElementException:
            driver.execute_script("window.scrollBy(0, 2000)")

        human_sleep(1.2, 2.0)

        if len(reviews_data) == last_count:
            no_new_count += 1
            if no_new_count >= 3:
                break
        else:
            no_new_count = 0
        last_count = len(reviews_data)

    if skipped_lang:
        log(f"     [WARNING] {skipped_lang} ta boshqa tildagi sharh o'tkazib yuborildi")

    return reviews_data

# ─────────────────────────────────────────
# ASOSIY FUNKSIYA
# ─────────────────────────────────────────
def scrape():
    os.makedirs("data", exist_ok=True)
    all_reviews = []
    seen_texts  = set()
    seen_places = set()

    # Eski yozilgan sharhlar va joy nomlarini yuklab olish (takrorlanishlarning oldini olish uchun)
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("Review_Text"):
                        seen_texts.add(row["Review_Text"])
                    if row.get("Place_Name"):
                        seen_places.add(row["Place_Name"])
        except Exception:
            pass

    log("=" * 55)
    log("RestoPulse - Google Maps Selenium Scraper")
    log(f"Qabul qilinadigan tillar: {ALLOWED_LANGS or 'barchasi'}")
    log("=" * 55)

    driver = make_driver()
    try:
        for query in SEARCH_QUERIES:
            query_str = query.strip()
            if query_str.startswith("http://") or query_str.startswith("https://"):
                log(f"\n[Direct URL] To'g'ridan-to'g'ri havola: '{query_str}'")
                place_urls = [query_str]
            else:
                log(f"\n[Search] Qidiruv: '{query_str}'")
                place_urls = search_places(driver, query_str)
                log(f"   {len(place_urls)} ta joy topildi")

            for i, url in enumerate(place_urls, 1):
                log(f"   [{i}/{len(place_urls)}] Yuklanmoqda...")
                # To'g'ridan-to'g'ri havola bo'lsa yoki SKIP_ALREADY_SCRAPED=False bo'lsa, skip logikasini chetlab o'tamiz
                is_direct = query_str.startswith("http://") or query_str.startswith("https://")
                skip_set = seen_places if (SKIP_ALREADY_SCRAPED and not is_direct) else None
                reviews = scrape_reviews(driver, url, skip_set)

                if reviews and reviews[0].get("skip"):
                    name = reviews[0]["Place_Name"]
                    log(f"   [Skip] {name} allaqachon skrap qilingan, o'tkazib yuboriladi.")
                    continue

                name = reviews[0]["Place_Name"] if reviews else "?"
                new  = 0
                for rv in reviews:
                    if rv["Review_Text"] not in seen_texts:
                        seen_texts.add(rv["Review_Text"])
                        all_reviews.append(rv)
                        new += 1

                log(f"   [OK] {name}: {new} ta sharh (jami: {len(all_reviews)})")
                seen_places.add(name)
                human_sleep(1.5, 3.0)
    finally:
        driver.quit()

    if all_reviews:
        fieldnames = ["Review_Text", "Sentiment", "Place_Name", "Google_Rating"]
        file_exists = os.path.exists(OUTPUT_FILE)
        with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(all_reviews)

        log("\n" + "=" * 55)
        log(f"[OK] TAYYOR!")
        log(f"   Jami sharh    : {len(all_reviews)}")
        log(f"   Fayl          : {OUTPUT_FILE}")

        from collections import Counter
        dist = Counter(rv["Sentiment"] for rv in all_reviews)
        log("\n   Yulduz taqsimoti:")
        for s in range(1, 6):
            cnt = dist.get(s, 0)
            stars_str = "*" * s + " " * (5 - s)
            log(f"   [{stars_str}] ({s} yulduz): {cnt:4d}")
        log("=" * 55)
    else:
        log("[WARNING] Hech qanday sharh topilmadi.")

    return all_reviews

if __name__ == "__main__":
    scrape()
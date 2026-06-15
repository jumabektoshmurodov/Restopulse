import os
import pandas as pd

def merge_datasets():
    main_file = "data/reviews.csv"
    google_file = "data/reviews_google.csv"
    
    if not os.path.exists(google_file):
        print(f"[ERROR] '{google_file}' fayli topilmadi. Avval skraperni ishga tushiring.")
        return
        
    if not os.path.exists(main_file):
        print(f"[WARNING] '{main_file}' topilmadi. Yangi fayl yaratiladi.")
        df_main = pd.DataFrame(columns=["Review_Text", "Sentiment"])
    else:
        df_main = pd.read_csv(main_file)
        
    # Google faylidan faqat kerakli ustunlarni olish
    df_google = pd.read_csv(google_file)
    if "Review_Text" not in df_google.columns or "Sentiment" not in df_google.columns:
        print("[ERROR] Google CSV faylida 'Review_Text' yoki 'Sentiment' ustunlari mavjud emas.")
        return
        
    df_google_clean = df_google[["Review_Text", "Sentiment"]]
    
    # Birlashtirish va dublikatlardan tozalash
    df_merged = pd.concat([df_main, df_google_clean], ignore_index=True)
    df_merged = df_merged.drop_duplicates(subset=["Review_Text"])
    
    # Saqlash
    os.makedirs("data", exist_ok=True)
    df_merged.to_csv(main_file, index=False, encoding="utf-8")
    
    print("=" * 50)
    print("[OK] DATASETLAR MUVAFFQAQIYATLI BIRLASHTIRILDI!")
    print(f"Eski ma'lumotlar soni: {len(df_main)} ta")
    print(f"Yangi ma'lumotlar soni: {len(df_google)} ta")
    print(f"Birlashtirilgan jami ma'lumotlar: {len(df_merged)} ta (dublikatlar olib tashlandi)")
    print(f"Fayl saqlangan manzil: {main_file}")
    print("=" * 50)

if __name__ == "__main__":
    merge_datasets()

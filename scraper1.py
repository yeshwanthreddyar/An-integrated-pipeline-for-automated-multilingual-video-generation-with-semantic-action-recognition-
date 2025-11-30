import os
import time
import shutil
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import WebDriverException
from webdriver_manager.chrome import ChromeDriverManager

# --- CONFIG ---
RAW_DIR = "data/raw"

def clear_old_files():
    """Remove old files in data/raw/"""
    if not os.path.exists(RAW_DIR):
        os.makedirs(RAW_DIR)
    else:
        print(f"Clearing old files from {RAW_DIR}/ ...")
        for f in os.listdir(RAW_DIR):
            file_path = os.path.join(RAW_DIR, f)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"⚠️ Error deleting {file_path}: {e}")

def get_driver():
    """Set up Chrome WebDriver with stable headless configuration"""
    print("🚀 Launching Chrome (headless)...")

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-software-rasterizer")
    options.add_argument("--log-level=3")

    # ✅ Use unique user-data-dir each time to prevent conflicts
    user_data_dir = os.path.join(os.getcwd(), "chrome_user_data", str(int(time.time())))
    os.makedirs(user_data_dir, exist_ok=True)
    options.add_argument(f"--user-data-dir={user_data_dir}")

    # ✅ Prevent session reuse crash
    options.add_argument("--remote-debugging-port=9222")

    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        return driver
    except WebDriverException as e:
        print("❌ Chrome WebDriver failed to start:")
        print(e)
        print("🔧 Try updating Chrome or ChromeDriver manually.")
        raise SystemExit(1)

def scrape_language(lang_code, lang_name):
    """Simulated scraping process per language (replace with real logic)"""
    print(f"\n🌐 Scraping language: {lang_code} ({lang_name})")
    try:
        driver = get_driver()
        driver.get("https://example.com")  # Replace with real target URL
        print("✅ Page loaded successfully!")
        time.sleep(2)
        driver.quit()
        print(f"✅ Scraping complete for {lang_code}")
    except Exception as e:
        print(f"⚠️ Error scraping {lang_code}: {e}")

def main():
    clear_old_files()

    languages = {
        "en": "English",
        "hi": "Hindi",
        "ur": "Urdu",
        "pa": "Punjabi",
        "gu": "Gujarati",
        "mr": "Marathi",
        "te": "Telugu",
        "kn": "Kannada",
        "ml": "Malayalam",
        "ta": "Tamil",
        "or": "Odia",
        "bn": "Bengali",
        "as": "Assamese",
        "mni": "Manipuri",
    }

    for lang_code, lang_name in languages.items():
        for attempt in range(3):
            try:
                scrape_language(lang_code, lang_name)
                break
            except Exception as e:
                print(f"⚠️ Attempt {attempt + 1}/3 failed for {lang_name}: {e}")
                time.sleep(2)
        else:
            print(f"❌ Failed to scrape {lang_name} after 3 attempts.")

if __name__ == "__main__":
    main()

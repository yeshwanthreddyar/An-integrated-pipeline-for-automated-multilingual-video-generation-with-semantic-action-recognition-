# scraper.py
import requests
import time
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path

# --- USE UNDETECTED_CHROMEDRIVER INSTEAD OF SELENIUM ---
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

OUT_DIR = Path("data/raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

LANG_MAP = {
    "en": "English", "hi": "Hindi", "ur": "Urdu", "pa": "Punjabi", "gu": "Gujarati",
    "mr": "Marathi", "te": "Telugu", "kn": "Kannada", "ml": "Malayalam", "tam,": "Tamil",
    "or": "Oriya", "bn": "Bengali", "as": "Assamese", "mni": "Manipuri"
}

def fetch_listing_for_language(lang_label):
    search_url = f"https://pib.gov.in/PressReleasePage.aspx?Language={lang_label}"
    
    options = uc.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument(f"user-agent={HEADERS['User-Agent']}")
    
    driver = None
    try:
        # --- Use uc.Chrome instead of webdriver.Chrome ---
        driver = uc.Chrome(options=options)
        
        print(f"  Navigating to {search_url} with undetected-chromedriver...")
        driver.get(search_url)
        
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.list-group a.list-group-item"))
        )
        
        return driver.page_source
    except Exception as e:
        print(f"  --> Error for {lang_label}: {e}")
        return None
    finally:
        if driver:
            driver.quit()

def extract_links_from_listing(html):
    if not html: return []
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.select("div.list-group a.list-group-item"):
        href = a.get('href')
        if href and "PRID=" in href.upper():
            full_url = requests.compat.urljoin("https://pib.gov.in/", href)
            links.append(full_url)
    if not links:
        print("--> No links found in the HTML source.")
    return list(dict.fromkeys(links))

def fetch_press_release(url):
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    title = soup.find("h2").get_text(strip=True) if soup.find("h2") else "No Title Found"
    date_div = soup.find("div", class_="ReleaseDate")
    date = date_div.get_text(strip=True) if date_div else "No Date Found"
    content_div = soup.find("div", id="pressrelease")
    body = content_div.get_text(separator="\n", strip=True) if content_div else ""
    ministry_div = soup.find("div", class_="Ministry")
    ministry = ministry_div.get_text(strip=True) if ministry_div else "No Ministry Found"
    return {"url": url, "title": title, "date": date, "text": body, "ministry": ministry}

def scrape_language(lang_code, max_articles=10, pause=1.0):
    lang_label = LANG_MAP.get(lang_code, lang_code)
    print(f"\nScraping language: {lang_code} ({lang_label})")
    html = fetch_listing_for_language(lang_label)
    if not html:
        print(f"Could not retrieve listing page for {lang_label}. Skipping.")
        return None
    links = extract_links_from_listing(html)
    print(f"Found {len(links)} links. Scraping up to {max_articles} articles.")
    if not links:
        df = pd.DataFrame(columns=["url", "title", "date", "text", "ministry", "language"])
    else:
        out = []
        for i, link in enumerate(links[:max_articles]):
            try:
                print(f"  Fetching article {i+1}/{min(len(links), max_articles)}: {link}")
                item = fetch_press_release(link)
                item['language'] = lang_code
                out.append(item)
                time.sleep(pause)
            except Exception as e:
                print(f"  --> Error fetching {link}: {e}")
        df = pd.DataFrame(out)
    out_file = OUT_DIR / f"pib_{lang_code}.csv"
    df.to_csv(out_file, index=False, encoding="utf-8-sig")
    if df.empty:
        print(f"⚠  Warning: No data was scraped for {lang_code}. Output file is empty.")
    else:
        print(f"✅ Saved {len(df)} articles to {out_file}")
    return out_file

if __name__ == "__main_":
    languages = [
        "en", "hi", "ur", "pa", "gu", "mr", "te", 
        "kn", "ml", "ta", "or", "bn", "as", "mni"
    ]
    print("Clearing old files from data/raw/...")
    for f in OUT_DIR.glob("*.csv"):
        f.unlink()
    for lang in languages:
        scrape_language(lang, max_articles=10, pause=0.5)
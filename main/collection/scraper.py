from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
import time
import csv
from pathlib import Path

def scrape_recommendations(seed_url, max_steps=50):
    BASE_DIR = Path(__file__).resolve().parent.parent
    OUTPUT_DIR = BASE_DIR / "data"
    OUTPUT_DIR.mkdir(exist_ok=True)

    OUTPUT_FILE = OUTPUT_DIR / "recommendation_walk.csv"
    options = Options()
    options.add_argument("--mute-audio")
    options.add_argument("--disable-notifications")
    options.add_argument("--start-maximized")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )

    wait = WebDriverWait(driver, 30)

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "title", "url", "description"])

    driver.get(seed_url)
    wait.until(EC.presence_of_element_located((By.TAG_NAME, "ytd-app")))
    time.sleep(4)

    current_url = seed_url

    for step in range(max_steps):
        print(f"\n--- STEP {step} ---")

        # Extract description from current page
        description = ""
        try:
            # Scroll down a bit to ensure description is loaded
            driver.execute_script("window.scrollBy(0, 300);")
            time.sleep(1)
            
            # Try to expand the description if it's collapsed
            try:
                expand_button = driver.find_element(By.CSS_SELECTOR, "tp-yt-paper-button#expand")
                driver.execute_script("arguments[0].click();", expand_button)
                time.sleep(0.5)
            except:
                pass  # Description might already be expanded or not collapsible
            
            # Extract the description text
            try:
                desc_element = driver.find_element(By.CSS_SELECTOR, "ytd-text-inline-expander#description-inline-expander yt-attributed-string")
                description = desc_element.text.strip()
            except:
                try:
                    # Fallback selector
                    desc_element = driver.find_element(By.CSS_SELECTOR, "#description-inline-expander")
                    description = desc_element.text.strip()
                except:
                    description = ""
            
            print(f"Description extracted: {len(description)} characters")
            
        except Exception as e:
            print(f"Could not extract description: {e}")
            description = ""

        # Scroll to load recommendations
        for _ in range(2):
            driver.execute_script("window.scrollBy(0, 800);")
            time.sleep(1)

        # 🔴 FALLBACK-BASED EXTRACTION (ROBUST)
        all_links = driver.find_elements(By.XPATH, "//a[@href]")
        candidates = []

        for link in all_links:
            try:
                href = link.get_attribute("href")
                title = link.get_attribute("title")
                aria = link.get_attribute("aria-label")
                text = link.text
            except:
                continue

            if not href or "watch?v=" not in href:
                continue
            if href == current_url:
                continue
            if "&lc=" in href:
                continue

            final_title = title or aria or text
            if not final_title:
                continue

            clean = final_title.strip().lower()

            # ❌ filter player controls
            if "next (shift+n)" in clean or clean == "next":
                continue

            # ❌ filter pure time labels
            if "minute" in clean and "second" in clean and len(clean) < 30:
                continue

            # ❌ filter timestamp links
            if "&t=" in href or "?t=" in href:
                continue

            # ❌ filter comment timestamps
            if "hour ago" in clean or "day ago" in clean:
                continue

            # ❌ very short / UI junk
            if len(final_title.strip()) < 15:
                continue

            candidates.append((final_title.strip(), href))


        if not candidates:
            print("No recommendations found. Retrying after wait...")
            time.sleep(5)

            all_links = driver.find_elements(By.XPATH, "//a[@href]")
            candidates = []

            for link in all_links:
                try:
                    href = link.get_attribute("href")
                    title = link.get_attribute("title")
                    aria = link.get_attribute("aria-label")
                    text = link.text
                except:
                    continue

                if not href or "watch?v=" not in href or href == current_url:
                    continue

                final_title = title or aria or text
                if final_title and len(final_title.strip()) >= 15:
                    candidates.append((final_title.strip(), href))

            if not candidates:
                print("Still no recommendations. Ending walk.")
                print(f"Walk ended early at step {step}")
                break

        title, next_url = candidates[0]

        print("TITLE:", title)
        print("URL:", next_url)

        with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([step, title, next_url, description])

        current_url = next_url
        driver.get(next_url)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "ytd-app")))
        time.sleep(3)

    driver.quit()
    print("\nDONE — recommendation walk complete")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        seed_url = sys.argv[1]
    else:
        seed_url = "https://www.youtube.com/watch?v=8nHBGFKLHZQ"
    scrape_recommendations(seed_url)
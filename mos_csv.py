import os
import csv

# Path to your generated video folder
GENERATED_DIR = "./generated"
OUTPUT_CSV = "mos_score.csv"

# Supported video formats
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")

def main():
    if not os.path.exists(GENERATED_DIR):
        raise FileNotFoundError(f"❌ Folder not found: {GENERATED_DIR}")

    videos = [
        f for f in os.listdir(GENERATED_DIR)
        if f.lower().endswith(VIDEO_EXTS)
    ]

    if not videos:
        print("⚠️ No video files found in generated/. Nothing to write.")
        return

    print(f"📁 Found {len(videos)} videos. Creating MOS CSV...")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "mos_score"])  # column headers

        for v in videos:
            writer.writerow([v, ""])  # Empty MOS scores

    print(f"✅ MOS CSV created: {OUTPUT_CSV}")
    print("👉 Fill the mos_score column manually after human evaluation.")

if __name__ == "__main__":
    main()

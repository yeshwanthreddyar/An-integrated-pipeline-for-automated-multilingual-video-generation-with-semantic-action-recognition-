# mos.py
import os
import csv
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from moviepy.editor import VideoFileClip

# ---------------------------------------
# CONFIG
# ---------------------------------------
GENERATED_DIR = "./generated"
MOS_CSV = "mos_score.csv"  # filename,mos_score


# ---------------------------------------
# CSV LOADING
# ---------------------------------------
def load_mos_scores(csv_path: str) -> Dict[str, List[float]]:
    """
    Load MOS scores from a CSV with at least:
        filename, mos_score

    - filename: video file name (e.g. generated_hi_115138.mp4)
    - mos_score: 1–5 (can be float)

    Returns: dict[filename] -> [scores...]
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ MOS CSV not found: {csv_path}")

    scores_by_video: Dict[str, List[float]] = defaultdict(list)

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        print(f"🧾 CSV columns detected in {csv_path}: {fieldnames}")

        if not fieldnames or "filename" not in fieldnames:
            raise ValueError(
                "❌ CSV must contain a 'filename' column.\n"
                "Example header:\n"
                "filename,mos_score"
            )

        # Detect MOS columns (any that start with 'mos')
        mos_cols = [c for c in fieldnames if c.lower().startswith("mos")]
        if not mos_cols:
            raise ValueError(
                "❌ No MOS columns detected.\n"
                "Add at least one column like 'mos_score' or 'mos_sync_r1'."
            )

        print(f"🎚 Using MOS columns: {mos_cols}")

        for row_idx, row in enumerate(reader, start=1):
            fname = row.get("filename", "").strip()
            if not fname:
                print(f"⚠️ Row {row_idx}: missing filename, skipping row.")
                continue

            for col in mos_cols:
                val = row.get(col, "").strip()
                if not val:
                    continue
                try:
                    score = float(val)
                    scores_by_video[fname].append(score)
                except ValueError:
                    print(
                        f"⚠️ Row {row_idx}, file '{fname}': invalid MOS value "
                        f"'{val}' in column '{col}'. Expected a number like 1–5."
                    )

    if not scores_by_video:
        raise ValueError(
            "❌ No valid MOS scores found in CSV.\n"
            "Check that:\n"
            "  • 'filename' column exists and matches your video filenames in ./generated\n"
            "  • At least one 'mos_*' column exists\n"
            "  • MOS cells contain numeric values like 1–5"
        )

    print(f"📥 Loaded MOS scores for {len(scores_by_video)} videos from {csv_path}")
    return scores_by_video


# ---------------------------------------
# VIDEO DURATION (ROBUST)
# ---------------------------------------
def get_video_duration(path: str) -> Optional[float]:
    """
    Return duration in seconds, or None if video is unreadable/corrupt.
    """
    if not os.path.exists(path):
        print(f"⚠️ Video not found on disk: {path}")
        return None

    try:
        with VideoFileClip(path) as clip:
            return float(clip.duration)
    except Exception as e:
        print(f"⚠️ Failed to read duration for '{path}': {e}")
        return None


# ---------------------------------------
# MAIN EVALUATION
# ---------------------------------------
def evaluate_mos_sync() -> None:
    """
    Compute:
      - Per-video MOS
      - Global average MOS
      - (Optional) basic A/V sync stats using duration (skips broken videos)
    """
    scores_by_video = load_mos_scores(MOS_CSV)

    # 1️⃣ Per-video MOS
    per_video_mos: List[Tuple[str, float]] = []
    for fname, scores in scores_by_video.items():
        if not scores:
            continue
        avg = sum(scores) / len(scores)
        per_video_mos.append((fname, avg))

    if not per_video_mos:
        raise ValueError("❌ No per-video MOS values could be computed.")

    # Sort by filename for stable display
    per_video_mos.sort(key=lambda x: x[0])

    print("\n🎥 Per-video MOS (subjective sync quality):")
    for fname, mos_val in per_video_mos:
        print(f"  • {fname}: MOS = {mos_val:.2f} (n={len(scores_by_video[fname])})")

    # 2️⃣ Global average MOS
    all_scores = [s for scores in scores_by_video.values() for s in scores]
    global_mos = sum(all_scores) / len(all_scores)
    print(f"\n⭐ Global Mean Opinion Score (MOS): {global_mos:.2f} over {len(all_scores)} ratings")

    # 3️⃣ Optional: check durations for sync sanity (skips broken videos)
    durations = []
    dur_fnames = []
    print("\n⏱ Checking video durations (for sync analysis)...")
    for fname, _ in per_video_mos:
        video_path = os.path.join(GENERATED_DIR, fname)
        dur = get_video_duration(video_path)
        if dur is None:
            print(f"   → Skipping '{fname}' from duration-based analysis.")
            continue
        durations.append(dur)
        dur_fnames.append(fname)
        print(f"   • {fname}: {dur:.2f} seconds")

    if durations:
        avg_dur = sum(durations) / len(durations)
        print(f"\n📊 Duration stats (on {len(durations)} readable videos):")
        print(f"   • Avg duration: {avg_dur:.2f} seconds")
        print(f"   • Min duration: {min(durations):.2f} seconds")
        print(f"   • Max duration: {max(durations):.2f} seconds")
    else:
        print("\n⚠️ No readable videos for duration analysis (but MOS stats are computed successfully).")


if __name__ == "__main__":
    evaluate_mos_sync()

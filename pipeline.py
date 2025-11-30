# run_pipeline.py
from pathlib import Path
from textpreprocess import split_into_sentences
# We still import MODEL_MAP, but we won't use it to get all keys
from translate import translate_sentences, MODEL_MAP 
from images import load_sd, generate_image_for_prompt, IMG_DIR
from assembly import make_scene_video

# --- Configuration ---
SAMPLE_TEXT = """The Union Minister said that the Government is committed to the welfare of farmers. A new scheme focusing on digital agriculture and crop insurance has been launched. This initiative will empower farmers with modern technology and provide financial security. Implementation will be monitored closely across all states."""

OUT_DIR = Path("data/demo_videos")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Helper Functions ---
def chunk_text_into_scenes(text, max_sentences=2):
    sents = split_into_sentences(text)
    scenes = [" ".join(sents[i:i+max_sentences]) for i in range(0, len(sents), max_sentences)]
    return scenes

def pipeline_for_text(text, target_lang, sd_pipe):
    lang_out_dir = OUT_DIR / target_lang
    lang_out_dir.mkdir(exist_ok=True)
    
    print(f"\n--- Processing video for target language: {target_lang.upper()} ---")
    
    scenes = chunk_text_into_scenes(text)
    print(f"📝 Split into {len(scenes)} scenes.")

    translated = translate_sentences(scenes, target_lang)
    print("✅ Translation complete.")

    image_files = []
    for i, scene_en in enumerate(scenes):
        prompt = f"A realistic news photograph of the Indian government announcing a new policy for farmer welfare and digital agriculture, photojournalism style."
        img_path = IMG_DIR / f"scene_{i}_{target_lang}.png"
        generate_image_for_prompt(sd_pipe, prompt, img_path)
        image_files.append(img_path)
    print("✅ Image generation complete.")

    video_files = []
    for i, (img, tr_text) in enumerate(zip(image_files, translated)):
        out_path = lang_out_dir / f"scene_{i}.mp4"
        make_scene_video(img, out_path, duration=7, subtitle_text=tr_text)
        video_files.append(out_path)
    print(f"✅ Scene assembly complete. Videos saved in '{lang_out_dir}'")
    
    return video_files

# --- Main Execution Block ---
if __name__ == "__main__":
    
    # --- THIS IS THE CHANGE ---
    # Instead of getting all languages, we define a specific list of the ones we want.
    languages_to_process = ["as", "hi", "ml", "mr"]
    
    print(f"🎯 Starting targeted video generation for {len(languages_to_process)} languages...")
    
    stable_diffusion_pipe = load_sd()
    
    # The loop will now only run for the languages in your new list
    for lang_code in languages_to_process:
        pipeline_for_text(SAMPLE_TEXT, lang_code, stable_diffusion_pipe)
        
    print("\n\n🎉 Targeted video generation is complete! 🎉")
# assemble.py
from moviepy.editor import ImageClip, VideoFileClip, concatenate_videoclips, TextClip, CompositeVideoClip
from pathlib import Path

VIDEO_DIR = Path("data/videos")
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

def make_scene_video(image_path, out_path, duration=5, subtitle_text=None):
    """Creates a silent video clip from an image and optional subtitle."""
    img_clip = ImageClip(str(image_path)).set_duration(duration)
    img_clip = img_clip.set_fps(24)

    if subtitle_text:
        txt_clip = TextClip(subtitle_text, fontsize=24, color='white', size=(img_clip.w - 40, None), method='caption')
        txt_clip = txt_clip.set_position(("center", img_clip.h - 80)).set_duration(duration)
        
        final_clip = CompositeVideoClip([img_clip, txt_clip])
        final_clip.write_videofile(str(out_path), fps=24, codec="libx264")
    else:
        img_clip.write_videofile(str(out_path), fps=24, codec="libx264")
        
    return out_path

def stitch_scene_videos(scene_files, out_path):
    """Stitches multiple video clips into one final video."""
    clips = [VideoFileClip(str(p)) for p in scene_files]
    final = concatenate_videoclips(clips)
    final.write_videofile(str(out_path), fps=24, codec="libx264", audio_codec="aac")
    return out_path

if __name__ == "__main__":
    # Example usage for a silent scene
    img = "data/images/demo1.png"
    out = VIDEO_DIR / "demo_scene_silent.mp4"
    make_scene_video(img, out, duration=7, subtitle_text="सरकार ने नई शिक्षा नीति शुरू की।")
    print("✅ Saved silent scene:", out)
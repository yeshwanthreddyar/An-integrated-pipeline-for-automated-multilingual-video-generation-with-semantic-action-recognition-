import os
import asyncio
import uuid
import numpy as np
import datetime
import random
from flask import Flask, render_template, request, send_from_directory
from moviepy.editor import ImageClip, concatenate_videoclips, TextClip, CompositeVideoClip
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from pydub import AudioSegment
from moviepy.audio.AudioClip import AudioArrayClip
import edge_tts

# Flask setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['VOICE_FOLDER'] = 'static/voiceover'
app.config['VIDEO_FOLDER'] = 'static/videos'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['VOICE_FOLDER'], exist_ok=True)
os.makedirs(app.config['VIDEO_FOLDER'], exist_ok=True)

# Language map
LANGUAGES = {
    "hi": "Hindi", "ta": "Tamil", "te": "Telugu", "bn": "Bengali",
    "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam",
    "mr": "Marathi", "or": "Odia", "pa": "Punjabi", "en": "English"
}

# ------------------------------------------------------------------
# Load translation model
# ------------------------------------------------------------------
print("⚙️ Loading universal translation model (facebook/m2m100_418M)...")
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")

def translate_text(text, lang_code):
    try:
        tokenizer.src_lang = "en"
        encoded = tokenizer(text, return_tensors="pt")
        generated = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id(lang_code))
        return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    except Exception as e:
        print(f"⚠️ Translation failed for {lang_code}: {e}")
        return text

# ------------------------------------------------------------------
# EDGE-TTS
# ------------------------------------------------------------------
async def tts_generate_async(text, path_wav, lang_code="hi"):
    voice_map = {
        "hi": "hi-IN-SwaraNeural", "ta": "ta-IN-PallaviNeural", "te": "te-IN-ShrutiNeural",
        "bn": "bn-IN-TanishaaNeural", "gu": "gu-IN-DhwaniNeural", "kn": "kn-IN-SapnaNeural",
        "ml": "ml-IN-MidhunNeural", "mr": "mr-IN-AarohiNeural", "or": "or-IN-KalpanaNeural",
        "pa": "pa-IN-AmritaNeural", "en": "en-IN-NeerjaNeural"
    }
    voice = voice_map.get(lang_code, "en-IN-NeerjaNeural")
    temp_mp3 = path_wav.replace(".wav", ".mp3")

    try:
        communicate = edge_tts.Communicate(text, voice=voice, rate="+90%", pitch="+5Hz")
        await communicate.save(temp_mp3)
        sound = AudioSegment.from_mp3(temp_mp3).set_channels(1).set_frame_rate(22050)
        sound.export(path_wav, format="wav")
        print(f"🎧 Generated neural TTS: {path_wav}")
    except Exception as e:
        print(f"⚠️ TTS failed: {e}")
    finally:
        if os.path.exists(temp_mp3):
            os.remove(temp_mp3)

def tts_generate(text, path_wav, lang_code="hi"):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(tts_generate_async(text, path_wav, lang_code))

# ------------------------------------------------------------------
# Cinematic Motion (Ken Burns)
# ------------------------------------------------------------------
def ken_burns_effect(image_path, duration):
    img_clip = ImageClip(image_path, duration=duration)
    zoom_factor = random.uniform(1.05, 1.15)
    move_x = random.randint(-40, 40)
    move_y = random.randint(-25, 25)
    clip = img_clip.resize(lambda t: 1 + (zoom_factor - 1) * (t / duration))
    clip = clip.set_position(lambda t: (move_x * t / duration, move_y * t / duration))
    return clip

# ------------------------------------------------------------------
# Generate Video
# ------------------------------------------------------------------
def create_video(images, text, lang_code):
    lang_name = LANGUAGES[lang_code]
    print(f"\n🌍 Generating cinematic video in {lang_name} ({lang_code})")

    sentences = [s.strip() for s in text.split('.') if s.strip()]
    scenes = list(zip(images, sentences))
    clips = []

    for idx, (img, sentence) in enumerate(scenes):
        translated = translate_text(sentence, lang_code)
        audio_path = os.path.join(app.config['VOICE_FOLDER'], f"{uuid.uuid4().hex}.wav")
        tts_generate(translated, audio_path, lang_code)

        # Load TTS
        sound = AudioSegment.from_wav(audio_path)
        samples = np.array(sound.get_array_of_samples()).astype(np.float32) / (2**15)
        if sound.channels == 1:
            samples = np.expand_dims(samples, axis=1)
        duration = len(sound) / 1000
        voice_clip = AudioArrayClip(samples, fps=sound.frame_rate).set_duration(duration)

        motion = ken_burns_effect(img, duration)
        subtitle = TextClip(
            translated, fontsize=38, color="white", method="caption",
            size=(motion.w - 200, None), align="center"
        ).set_duration(duration).set_position(("center", motion.h - 120))

        final_scene = CompositeVideoClip([motion.set_audio(voice_clip), subtitle])
        clips.append(final_scene.crossfadein(0.8))

    if not clips:
        raise Exception("No valid clips generated")

    final_video = concatenate_videoclips(clips, method="compose", padding=-0.5)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(app.config['VIDEO_FOLDER'], f"cinematic_{lang_code}_{timestamp}.mp4")

    final_video.write_videofile(
        output_path, fps=30, codec="libx264", audio_codec="aac",
        ffmpeg_params=["-profile:v", "baseline", "-level", "3.0", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    )

    return output_path

# ------------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html', languages=LANGUAGES)

@app.route('/generate', methods=['POST'])
def generate():
    text = request.form.get('text')
    lang_code = request.form.get('language')
    files = request.files.getlist('images')

    if not text or not files:
        return "Please upload images and enter text."

    image_paths = []
    for file in files:
        fname = f"{uuid.uuid4().hex}.png"
        path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        file.save(path)
        image_paths.append(path)

    output_path = create_video(image_paths, text, lang_code)
    video_file = os.path.basename(output_path)
    return render_template('result.html', video_file=video_file)

@app.route('/videos/<filename>')
def download(filename):
    return send_from_directory(app.config['VIDEO_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)

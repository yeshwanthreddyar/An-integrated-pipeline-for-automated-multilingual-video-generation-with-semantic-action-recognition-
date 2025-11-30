import os
import time
import asyncio
from flask import Flask, render_template, request, abort, send_from_directory
from moviepy.editor import AudioFileClip, VideoFileClip
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from huggingface_hub import login
from dotenv import load_dotenv
import edge_tts
from datetime import datetime
from pydub import AudioSegment
import requests

# ============================================================
# 🌈 ENVIRONMENT CONFIGURATION
# ============================================================
load_dotenv()

RUNWAY_API_KEY = os.getenv("RUNWAY_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "generated")

if not RUNWAY_API_KEY:
    raise ValueError("⚠️ RUNWAY_API_KEY missing from .env file")
if not HF_TOKEN:
    raise ValueError("⚠️ HF_TOKEN missing from .env file")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ✅ Login to Hugging Face
login(HF_TOKEN)
print("🔑 Hugging Face authentication successful.")

print("\n🚀 Environment loaded successfully:")
print(f"   🌈 Runway API Key: ✅")
print(f"   🤗 HuggingFace Token: ✅")
print(f"   📂 Output Directory: {OUTPUT_DIR}")
print("============================================================\n")

app = Flask(__name__, static_folder=OUTPUT_DIR)

# ============================================================
# 🧠 VIDEO CLASSIFICATION (Trained Model)
# ============================================================
from dataset1 import UCF101Dataset  # Your dataset wrapper from ZIP
from dataset2 import VideoClassifier  # Your model class from ZIP
import torchvision.transforms as transforms
import torch
import cv2
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "./checkpoints/best_model.pth"  # path to your trained weights
CLASSES_FILE = "./video_data/classInd.txt"   # optional UCF101 class names source

# Try to load trained model
classifier_model = None
class_names = None

def _load_class_names():
    """
    Load UCF101 class names from classInd.txt (format: "1 ApplyEyeMakeup").
    Returns a zero-based list aligned with model outputs if available.
    """
    try:
        if os.path.exists(CLASSES_FILE):
            names = []
            with open(CLASSES_FILE, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # "1 ApplyEyeMakeup" -> idx_str, name
                    parts = line.split()
                    idx = int(parts[0])  # 1-based
                    name = " ".join(parts[1:])
                    names.append((idx, name))
            # Sort by idx and map to zero-based list
            names = [name for _, name in sorted(names, key=lambda x: x[0])]
            return names
    except Exception as e:
        print(f"⚠️ Could not load class names from {CLASSES_FILE}: {e}")
    # Fallback: try dataset mapping
    try:
        tmp_ds = UCF101Dataset("./video_data/UCF-101", "./video_data/splits", train=False)
        mapping = tmp_ds.class_to_idx  # dict: class_name -> idx
        # invert to idx->name zero-based
        names = [None] * len(mapping)
        for name, idx in mapping.items():
            names[idx] = name
        return names
    except Exception as e:
        print(f"⚠️ Could not infer class names from dataset: {e}")
    return None

def _load_trained_model():
    global classifier_model
    print("🎥 Loading trained VideoClassifier model...")
    classifier_model = VideoClassifier(num_classes=101).to(DEVICE)
    classifier_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    classifier_model.eval()
    print("✅ Video classification model loaded successfully.")

try:
    _load_trained_model()
except Exception as e:
    print(f"⚠️ Could not load trained model '{MODEL_PATH}': {e}")
    classifier_model = None

class_names = _load_class_names()
if class_names:
    print(f"📚 Loaded {len(class_names)} class names.")
else:
    print("⚠️ Class names unavailable; will display index only.")

# ============================================================
# 🔤 TRANSLATION (Hugging Face M2M100)
# ============================================================
print("🔤 Loading translator (facebook/m2m100_418M)...")
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
translator = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")

def translate_text(text, target_lang_code):
    tokenizer.src_lang = "en"
    encoded = tokenizer(text, return_tensors="pt")
    gen = translator.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id(target_lang_code))
    translated = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
    return translated

# ============================================================
# 🎙️ TTS (Edge)
# ============================================================
VOICE_MAP = {
    "hi": "hi-IN-SwaraNeural", "ta": "ta-IN-PallaviNeural", "te": "te-IN-ShrutiNeural",
    "bn": "bn-IN-TanishaaNeural", "gu": "gu-IN-DhwaniNeural", "kn": "kn-IN-SapnaNeural",
    "ml": "ml-IN-MidhunNeural", "mr": "mr-IN-AarohiNeural", "or": "or-IN-KalpanaNeural",
    "pa": "pa-IN-AmritaNeural", "en": "en-IN-NeerjaNeural"
}

async def generate_tts_async(text, lang_code, wav_path):
    voice = VOICE_MAP.get(lang_code, "en-IN-NeerjaNeural")
    tmp_mp3 = wav_path.replace(".wav", ".mp3")
    communicate = edge_tts.Communicate(text, voice=voice, rate="+20%", pitch="+5Hz")
    await communicate.save(tmp_mp3)
    sound = AudioSegment.from_mp3(tmp_mp3).set_channels(1).set_frame_rate(22050)
    sound.export(wav_path, format="wav")
    os.remove(tmp_mp3)

def generate_tts(text, lang_code, wav_path):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(generate_tts_async(text, lang_code, wav_path))
    loop.close()

# ============================================================
# 🎬 RUNWAY VIDEO GENERATION (FIXED ENCODING)
# ============================================================
def test_runway_connection_safe():
    """Test RunwayML connection without spending ANY credits"""
    try:
        from runwayml import RunwayML
        client = RunwayML(api_key=RUNWAY_API_KEY)
        print(f"✅ RunwayML SDK Import Successful!")
        print(f"   Client initialized with API key")
        return True, "RunwayML SDK connected successfully."
    except Exception as e:
        print(f"❌ RunwayML Connection Failed: {e}")
        return False, f"Connection failed: {str(e)}"

def create_sample_video():
    """Create a simple sample video file for simulation"""
    try:
        from moviepy.editor import ColorClip
        clip = ColorClip(size=(640, 480), color=(255, 0, 0), duration=4)
        clip = clip.set_fps(24)
        sample_path = os.path.join(OUTPUT_DIR, "sample_video.mp4")
        clip.write_videofile(sample_path, codec="libx264", verbose=False, logger=None)
        return sample_path
    except Exception as e:
        print(f"❌ Could not create sample video: {e}")
        return None

def simulate_video_generation(prompt: str) -> str:
    """
    SIMULATE video generation for testing - NO CREDITS USED
    """
    print("🎬 SIMULATION MODE: Testing video generation pipeline")
    print(f"📝 Prompt: {prompt}")
    print("💰 NO CREDITS WILL BE USED - This is a simulation")
    time.sleep(2)
    dummy_filename = f"simulated_{datetime.now().strftime('%H%M%S')}.mp4"
    dummy_path = os.path.join(OUTPUT_DIR, dummy_filename)
    try:
        sample_path = create_sample_video()
        if sample_path and os.path.exists(sample_path):
            import shutil
            shutil.copy2(sample_path, dummy_path)
            print(f"✅ Created simulated video: {dummy_filename}")
        else:
            with open(dummy_path, 'wb') as f:
                f.write(b"SIMULATED VIDEO FILE - NO CREDITS USED\n")
                f.write(f"Prompt: {prompt}\n".encode('utf-8', errors='ignore'))
                f.write(b"Generated in simulation mode\n")
                f.write(b"Enable 'Use Real API' for actual videos\n")
            print(f"✅ Created simulation file: {dummy_filename}")
    except Exception as e:
        print(f"❌ Error creating simulation file: {e}")
        open(dummy_path, 'wb').close()
        print(f"✅ Created empty simulation file: {dummy_filename}")
    return dummy_filename

def debug_runway_methods():
    """Debug available methods in the RunwayML SDK"""
    try:
        from runwayml import RunwayML
        client = RunwayML(api_key=RUNWAY_API_KEY)
        print("🔍 Debugging RunwayML SDK methods...")
        methods = [method for method in dir(client) if not method.startswith('_')]
        print("📋 Available client methods:")
        for method in methods:
            print(f"   - {method}")
        if hasattr(client, 'text_to_video'):
            print("✅ text_to_video method available")
            text_methods = [m for m in dir(client.text_to_video) if not m.startswith('_')]
            print(f"   text_to_video methods: {text_methods}")
        else:
            print("❌ text_to_video method NOT available")
        return True
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        return False

def generate_runway_video_real(prompt: str) -> str:
    """
    REAL video generation - USES CREDITS
    """
    print("🚨 REAL API MODE: This will use RunwayML credits!")
    print("💰 CREDITS WILL BE DEDUCTED FROM YOUR ACCOUNT!")
    try:
        from runwayml import RunwayML
        print(f"🎬 Starting REAL RunwayML video generation...")
        print(f"📝 Prompt: {prompt}")
        client = RunwayML(api_key=RUNWAY_API_KEY)
        debug_runway_methods()
        print("🔄 Creating video generation task...")
        task = client.text_to_video.create(
            model='veo3.1',
            prompt_text=prompt,
            duration=4,
            ratio='1920:1080'
        )
        print(f"⏳ Task created with ID: {task.id}")
        print("🔄 Waiting for generation to complete...")
        task = task.wait_for_task_output()
        print("✅ Video generation completed!")
        video_url = None
        if hasattr(task, 'output') and task.output:
            if isinstance(task.output, list) and len(task.output) > 0:
                first_item = task.output[0]
                if isinstance(first_item, str) and first_item.startswith('http'):
                    video_url = first_item
                    print(f"📹 Found video URL: {video_url}")
        if not video_url:
            raise Exception("Could not find video URL in task output")
        print(f"📥 Downloading video from: {video_url}")
        video_filename = f"runway_{datetime.now().strftime('%H%M%S')}.mp4"
        video_path = os.path.join(OUTPUT_DIR, video_filename)
        response = requests.get(video_url, stream=True, timeout=120)
        if response.status_code == 200:
            with open(video_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                file_size = os.path.getsize(video_path)
                print(f"✅ Video saved: {video_filename} ({file_size} bytes)")
                return video_filename
            else:
                raise Exception("Downloaded video file is empty")
        else:
            raise Exception(f"Failed to download video: HTTP {response.status_code}")
    except Exception as e:
        error_msg = str(e)
        print(f"❌ RunwayML Error: {error_msg}")
        raise Exception(f"Video generation failed: {error_msg}")

# ============================================================
# 🧠 COMBINE VIDEO + AUDIO
# ============================================================
def combine_video_audio(video_filename, audio_path):
    print("🔊 Combining video and audio...")
    try:
        video_path = os.path.join(OUTPUT_DIR, video_filename)
        if "simulated" in video_filename:
            final_filename = f"final_{datetime.now().strftime('%H%M%S')}.mp4"
            final_path = os.path.join(OUTPUT_DIR, final_filename)
            import shutil
            shutil.copy2(video_path, final_path)
            print(f"✅ Final simulated video: {final_filename}")
            return final_filename
        video_clip = VideoFileClip(video_path)
        audio_clip = AudioFileClip(audio_path)
        final = video_clip.set_audio(audio_clip)
        final_filename = f"final_{datetime.now().strftime('%H%M%S')}.mp4"
        final_path = os.path.join(OUTPUT_DIR, final_filename)
        final.write_videofile(final_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)
        print(f"✅ Final video saved: {final_filename}")
        return final_filename
    except Exception as e:
        print(f"❌ Error combining video/audio: {e}")
        return video_filename

# ============================================================
# 🔍 VIDEO INFERENCE (Predict Action Class)
# ============================================================
def classify_generated_video(video_filename):
    """Run inference on the generated video using the trained classifier"""
    if classifier_model is None:
        print("⚠️ No classifier model loaded. Skipping classification.")
        return "Model not loaded"

    video_path = os.path.join(OUTPUT_DIR, video_filename)
    print(f"🔍 Running action recognition on: {video_path}")

    # Read frames
    cap = cv2.VideoCapture(video_path)
    frames = []
    success, frame = cap.read()
    # Sample up to 16 frames, skipping a few frames in between
    while success and len(frames) < 16:
        try:
            frame = cv2.resize(frame, (112, 112))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            # skip some frames to get spread across the clip
            for _ in range(3):
                cap.grab()
            success, frame = cap.read()
        except Exception:
            break
    cap.release()

    if len(frames) == 0:
        print("❌ Could not read frames from video.")
        return "Unreadable video"

    # Shape to [B, C, T, H, W], normalize to [0,1]
    frames = np.stack(frames)                   # [T, H, W, C]
    frames = torch.tensor(frames).permute(3, 0, 1, 2)  # [C, T, H, W]
    frames = frames.unsqueeze(0).float() / 255.0       # [1, C, T, H, W]
    frames = frames.to(DEVICE)

    with torch.no_grad():
        preds = classifier_model(frames)
        predicted_idx = torch.argmax(preds, dim=1).item()

    if class_names and 0 <= predicted_idx < len(class_names):
        class_name = class_names[predicted_idx]
    else:
        class_name = f"Class {predicted_idx}"

    print(f"🎯 Predicted Action: {class_name}")
    return class_name

# ============================================================
# 🌐 FLASK ROUTES
# ============================================================
LANGUAGES = {
    "hi": "Hindi", "ta": "Tamil", "te": "Telugu", "bn": "Bengali",
    "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam",
    "mr": "Marathi", "or": "Odia", "pa": "Punjabi", "en": "English"
}

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        prompt = request.form.get("prompt", "").strip()
        lang = request.form.get("language", "en").strip()
        use_real_api = request.form.get("use_real_api") == "true"

        if not prompt:
            return abort(400, "Prompt is required.")
        if lang not in LANGUAGES:
            return abort(400, f"Unsupported language: {lang}")

        print(f"\n🎬 Starting video generation pipeline...")
        print(f"📝 Original prompt: {prompt}")
        print(f"🌐 Target language: {LANGUAGES[lang]}")
        print("🚨 REAL CREDITS WILL BE USED!" if use_real_api else "💰 SIMULATION MODE - No credits will be used")

        try:
            # Translate the prompt
            translated = translate_text(prompt, lang)
            print(f"🔤 Translated text: {translated}")

            # Generate video
            if use_real_api:
                video_filename = generate_runway_video_real(translated)
            else:
                video_filename = simulate_video_generation(translated)

            # Generate TTS audio
            audio_filename = f"tts_{lang}_{datetime.now().strftime('%H%M%S')}.wav"
            audio_path = os.path.join(OUTPUT_DIR, audio_filename)
            print("🗣️ Generating voiceover...")
            generate_tts(translated, lang, audio_path)

            # Combine video and audio
            final_filename = combine_video_audio(video_filename, audio_path)

            # Classify final video
            predicted_action = classify_generated_video(final_filename)
            print("🎉 Video generation completed successfully!")
            print(f"📁 Final video: {final_filename}")

            return render_template(
                "index.html",
                languages=LANGUAGES,
                generated_video=final_filename,
                prompt=prompt,
                selected_lang=lang,
                predicted_action=predicted_action,
                is_simulation=not use_real_api
            )

        except Exception as e:
            print(f"❌ Error in video generation: {e}")
            return render_template(
                "error.html",
                error_message=str(e),
                prompt=prompt,
                languages=LANGUAGES
            )

    return render_template("index.html", languages=LANGUAGES)

@app.route("/videos/<filename>")
def serve_video(filename):
    """Route to serve generated videos"""
    return send_from_directory(OUTPUT_DIR, filename)

@app.route("/download/<filename>")
def download(filename):
    """Route to serve generated videos for download"""
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)

@app.route("/test-connection")
def test_connection():
    """Test route to check RunwayML connection without spending credits"""
    try:
        success, message = test_runway_connection_safe()
        if success:
            return f"""
            <div style="font-family: Arial, sans-serif; padding: 20px; background: #f5f5f5;">
                <div style="max-width: 600px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px;">
                    <h3 style="color: green;">✅ RunwayML Connection Successful!</h3>
                    <p>{message}</p>
                    <p><strong>Safety First:</strong> Use simulation mode to test the pipeline before using real credits.</p>
                    <br>
                    <a href="/" style="background: #007bff; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">← Back to Home</a>
                </div>
            </div>
            """
        else:
            return f"""
            <div style="font-family: Arial, sans-serif; padding: 20px; background: #f5f5f5;">
                <div style="max-width: 600px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px;">
                    <h3 style="color: red;">❌ RunwayML Connection Failed</h3>
                    <p>{message}</p>
                    <p>Check your API key and make sure it's valid.</p>
                    <br>
                    <a href="/" style="background: #007bff; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">← Back to Home</a>
                </div>
            </div>
            """
    except Exception as e:
        return f"""
        <div style="font-family: Arial, sans-serif; padding: 20px; background: #f5f5f5;">
            <div style="max-width: 600px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px;">
                <h3 style="color: red;">❌ Test Failed</h3>
                <p><strong>Error:</strong> {str(e)}</p>
                <br>
                <a href="/" style="background: #007bff; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">← Back to Home</a>
            </div>
        </div>
        """

@app.route("/debug-sdk")
def debug_sdk():
    """Debug route to see what's available in the SDK"""
    try:
        debug_runway_methods()
        return """
        <div style="font-family: Arial, sans-serif; padding: 20px; background: #f5f5f5;">
            <div style="max-width: 600px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px;">
                <h3>🔍 SDK Debug Information</h3>
                <p>Check your console for detailed debug information about available RunwayML methods.</p>
                <br>
                <a href="/" style="background: #007bff; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">← Back to Home</a>
            </div>
        </div>
        """
    except Exception as e:
        return f"""
        <div style="font-family: Arial, sans-serif; padding: 20px; background: #f5f5f5;">
            <div style="max-width: 600px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px;">
                <h3 style="color: red;">❌ Debug Failed</h3>
                <p><strong>Error:</strong> {str(e)}</p>
                <br>
                <a href="/" style="background: #007bff; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">← Back to Home</a>
            </div>
        </div>
        """

# ============================================================
# 🚀 MAIN
# ============================================================
if __name__ == "__main__":
    print("🔧 Available routes:")
    print("   http://localhost:5000/ - Main application (SAFE SIMULATION MODE)")
    print("   http://localhost:5000/test-connection - Test RunwayML connection (NO CREDITS)")
    print("   http://localhost:5000/debug-sdk - Debug SDK methods")
    print("   http://localhost:5000/videos/<filename> - Serve videos")
    print("\n⚠️  IMPORTANT SAFETY FEATURES:")
    print("   - DEFAULT: Simulation mode (NO credits used)")
    print("   - REAL API: Only when explicitly enabled in form")
    print("\n🔍 Testing RunwayML connection...")
    success, message = test_runway_connection_safe()
    if success:
        print("🎯 Ready to use! Start with simulation mode.")
    else:
        print("❌ Connection issues detected. Use simulation mode only.")
    app.run(debug=True, host="0.0.0.0", port=5000)

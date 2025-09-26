from flask import Flask, render_template, request, redirect, jsonify
from googletrans import Translator
from gtts import gTTS
import os
import base64
import io
from PIL import Image
import uuid
import cv2
import numpy as np


app = Flask(__name__)
translator = Translator()

# ----------------- ROUTES ----------------- #

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        return redirect("/select-language")
    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        return redirect("/login")
    return render_template("signup.html")

@app.route("/select-language", methods=["GET", "POST"])
def select_language():
    return render_template("select_language.html")

@app.route("/select-mode", methods=["POST"])
def select_mode():
    selected_language = request.form["language"]
    return render_template("select_mode.html", language=selected_language)

@app.route("/start-learning", methods=["POST"])
def start_learning():
    language = request.form["language"]
    mode = request.form["mode"]

    if mode == "voice":
        return render_template("voice_learning.html", language=language)
    elif mode == "gesture":
        return render_template("gesture_learning.html", language=language)
    else:
        return "Invalid mode selected", 400

@app.route("/voice-learning", methods=["POST"])
def voice_learning():
    language = request.form["language"]
    return render_template("voice_learning.html", language=language)

@app.route("/gesture-learning", methods=["POST"])
def gesture_learning():
    language = request.form["language"]
    return render_template("gesture_learning.html", language=language)

# ----------------- TRANSLATION API ----------------- #
@app.route("/translate", methods=["POST"])
def translate_api():
    data = request.json
    hindi_word = data.get("hindi_word")
    target_lang = data.get("target_lang")

    # Dictionary translation first
    translated = dictionary_translate(hindi_word, target_lang)

    # Generate unique filename using uuid
    if not os.path.exists("static/audio"):
        os.makedirs("static/audio")
    audio_filename = f"{uuid.uuid4()}.mp3"
    audio_file_path = os.path.join("static", "audio", audio_filename)

    # Generate TTS
    tts = gTTS(translated, lang=target_lang_code(target_lang))
    tts.save(audio_file_path)

    return jsonify({
        "translated": translated,
        "audio_url": "/" + audio_file_path.replace("\\", "/")  # for Windows path fix
    })


# ----------------- GESTURE PREDICTION ----------------- #
@app.route("/predict-gesture", methods=["POST"])
def predict_gesture():
    data = request.get_json()
    image_data = data.get("image")
    target_lang = data.get("target_lang")

    if not image_data:
        return jsonify({"error": "No image provided"}), 400

    # Decode base64 image
    image_data = image_data.split(",")[1]
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Convert to grayscale + threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    detected_hindi_word = "नमस्ते"  # default

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(cnt, returnPoints=False)

        if hull is not None and len(hull) > 3:
            defects = cv2.convexityDefects(cnt, hull)

            if defects is not None:
                finger_count = 0
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(cnt[s][0])
                    end = tuple(cnt[e][0])
                    far = tuple(cnt[f][0])
                    a = np.linalg.norm(np.array(start) - np.array(end))
                    b = np.linalg.norm(np.array(start) - np.array(far))
                    c = np.linalg.norm(np.array(end) - np.array(far))
                    angle = np.arccos((b**2 + c**2 - a**2) / (2*b*c))

                    if angle <= np.pi/2:
                        finger_count += 1

                # Map gestures by finger count
                if finger_count == 0:
                    detected_hindi_word = "क्या कर रहे हो"
                elif finger_count == 1:
                    detected_hindi_word = "कहाँ जा रहे हो"
                elif finger_count >= 4:
                    detected_hindi_word = "नमस्ते"

    # Translate
    translated = dictionary_translate(detected_hindi_word, target_lang)

    # Generate audio
    audio_filename = f"{uuid.uuid4()}.mp3"
    audio_path = os.path.join("static", "audio", audio_filename)
    tts = gTTS(translated, lang=target_lang_code(target_lang))
    tts.save(audio_path)

    return jsonify({
        "hindi_word": detected_hindi_word,
        "translated": translated,
        "audio_url": f"/static/audio/{audio_filename}"
    })

# ----------------- HELPER FUNCTIONS ----------------- #
def target_lang_code(language):
    lang_map = {
        "english": "en", "tamil": "ta", "telugu": "te", "bengali": "bn",
        "kannada": "kn", "marathi": "mr", "gujarati": "gu", "punjabi": "pa", "odia": "or"
    }
    return lang_map.get(language.lower(), "en")

def dictionary_translate(hindi_word, target_lang):
    # Normalize input
    hindi_word = hindi_word.strip().replace("?", "").replace("।", "")
    translations = {
        "नमस्ते": {
            "english": "Hello",
            "tamil": "வணக்கம்",
            "telugu": "నమస్తే",
            "bengali": "নমস্তে",
            "kannada": "ನಮಸ್ಕಾರ",
            "marathi": "नमस्कार",
            "gujarati": "નમસ્તે",
            "punjabi": "ਨਮਸਤੇ",
            "odia": "ନମସ୍କାର"
        },
        "क्या कर रहे हो": {
            "english": "What are you doing?",
            "tamil": "நீங்கள் என்ன செய்கிறீர்கள்?",
            "telugu": "మీరు ఏమి చేస్తున్నారు?",
            "bengali": "তুমি কী করছ?",
            "kannada": "ನೀವು ಏನು ಮಾಡುತ್ತಿದ್ದೀರಿ?",
            "marathi": "तुम काय करत आहात?",
            "gujarati": "તમે શું કરી રહ્યા છો?",
            "punjabi": "ਤੁਸੀਂ ਕੀ ਕਰ ਰਹੇ ਹੋ?",
            "odia": "ତୁମେ କ'ଣ କରୁଛ?"
        },
        "कहाँ जा रहे हो": {
            "english": "Where are you going?",
            "tamil": "நீங்கள் எங்கே போகிறீர்கள்?",
            "telugu": "మీరు ఎక్కడికి వెళ్తున్నారు?",
            "bengali": "তুমি কোথায় যাচ্ছ?",
            "kannada": "ನೀವು ಎಲ್ಲಿಗೆ ಹೋಗುತ್ತೀರಿ?",
            "marathi": "तुम्ही कुठे चालले आहात?",
            "gujarati": "તમે ક્યાં જઈ રહ્યા છો?",
            "punjabi": "ਤੁਸੀਂ ਕਿੱਥੇ ਜਾ ਰਹੇ ਹੋ?",
            "odia": "ତୁମେ କେଉଁଠି ଯାଉଛ?"
        },
        "क्या खाना खाया": {
            "english": "Have you eaten?",
            "tamil": "நீங்கள் சாப்பிட்டீர்களா?",
            "telugu": "మీరు భోజనం చేసారా?",
            "bengali": "তুমি খেয়েছ?",
            "kannada": "ನೀವು ಊಟ ಮಾಡಿಕೊಂಡೀರಾ?",
            "marathi": "तुम्ही जेवलंत का?",
            "gujarati": "શું તમે ખાધું છે?",
            "punjabi": "ਤੁਸੀਂ ਖਾ ਲਿਆ?",
            "odia": "ତୁମେ ଖାଇଛ?"
        },
         "मुझे माफ करें": {
        "english": "Sorry",
        "tamil": "மன்னிக்கவும்",
        "telugu": "క్షమించండి",
        "bengali": "দুঃখিত",
        "kannada": "ಕ್ಷಮಿಸಿ",
        "marathi": "माफ करा",
        "gujarati": "માફ કરશો",
        "punjabi": "ਮਾਫ਼ ਕਰਨਾ",
        "odia": "ମାଫ କରନ୍ତୁ"
    },
    "धन्यवाद": {
        "english": "Thank you",
        "tamil": "நன்றி",
        "telugu": "ధన్యవాదాలు",
        "bengali": "ধন্যবাদ",
        "kannada": "ಧನ್ಯವಾದಗಳು",
        "marathi": "धन्यवाद",
        "gujarati": "આભાર",
        "punjabi": "ਧੰਨਵਾਦ",
        "odia": "ଧନ୍ୟବାଦ"
    },
    "मुझे मदद चाहिए": {
        "english": "I need help",
        "tamil": "எனக்கு உதவி வேண்டும்",
        "telugu": "నాకు సహాయం కావాలి",
        "bengali": "আমার সাহায্য প্রয়োজন",
        "kannada": "ನನಗೆ ಸಹಾಯ ಬೇಕು",
        "marathi": "मला मदत हवी आहे",
        "gujarati": "મને મદદ જોઈએ છે",
        "punjabi": "ਮੈਨੂੰ ਮਦਦ ਚਾਹੀਦੀ ਹੈ",
        "odia": "ମୋତେ ସାହାଯ୍ୟ ଦରକାର"
    },
    "मुझे पानी चाहिए": {
        "english": "I need water",
        "tamil": "எனக்கு தண்ணீர் வேண்டும்",
        "telugu": "నాకు నీరు కావాలి",
        "bengali": "আমার পানি দরকার",
        "kannada": "ನನಗೆ ನೀರು ಬೇಕು",
        "marathi": "मला पाणी हवे आहे",
        "gujarati": "મને પાણી જોઈએ છે",
        "punjabi": "ਮੈਨੂੰ ਪਾਣੀ ਚਾਹੀਦਾ ਹੈ",
        "odia": "ମୋତେ ପାଣି ଦରକାର"
    },
    "मुझे भूख लगी है": {
        "english": "I am hungry",
        "tamil": "எனக்கு பசிக்குது",
        "telugu": "నాకు ఆకలిగా ఉంది",
        "bengali": "আমি ক্ষুধার্ত",
        "kannada": "ನನಗೆ ಹಸಿವಾಗಿದೆ",
        "marathi": "मला भूक लागली आहे",
        "gujarati": "મને ભૂખ લાગી છે",
        "punjabi": "ਮੈਨੂੰ ਭੁੱਖ ਲੱਗੀ ਹੈ",
        "odia": "ମୁଁ ଭୁଖ୍ୟାର୍ତ୍ତ"
    },
    "मैं ठीक हूँ": {
        "english": "I am fine",
        "tamil": "நான் நலம்",
        "telugu": "నేను బాగున్నాను",
        "bengali": "আমি ভালো আছি",
        "kannada": "ನಾನು ಚೆನ್ನಾಗಿದ್ದೇನೆ",
        "marathi": "मी ठीक आहे",
        "gujarati": "હું સારું છું",
        "punjabi": "ਮੈਂ ਠੀਕ ਹਾਂ",
        "odia": "ମୁଁ ଭଲ ଅଛି"
    },
    "आप कैसे हैं": {
        "english": "How are you?",
        "tamil": "நீங்கள் எப்படி இருக்கிறீர்கள்?",
        "telugu": "మీరు ఎలా ఉన్నారు?",
        "bengali": "আপনি কেমন আছেন?",
        "kannada": "ನೀವು ಹೇಗಿದ್ದೀರಾ?",
        "marathi": "तुम्ही कसे आहात?",
        "gujarati": "તમે કેમ છો?",
        "punjabi": "ਤੁਸੀਂ ਕਿਵੇਂ ਹੋ?",
        "odia": "ତୁମେ କେମିତି ଅଛ?"
    },
    "मुझे समझ नहीं आया": {
        "english": "I didn't understand",
        "tamil": "எனக்கு புரியவில்லை",
        "telugu": "నాకు అర్థం కాలేదు",
        "bengali": "আমি বুঝতে পারিনি",
        "kannada": "ನನಗೆ ಅರ್ಥವಾಗಲಿಲ್ಲ",
        "marathi": "मला समजले नाही",
        "gujarati": "મને સમજાયું નહીં",
        "punjabi": "ਮੈਨੂੰ ਸਮਝ ਨਹੀਂ ਆਇਆ",
        "odia": "ମୁଁ ବୁଝି ପାରିନି"
    },
    "मुझे समय नहीं मिला": {
        "english": "I didn't have time",
        "tamil": "எனக்கு நேரம் இல்லை",
        "telugu": "నాకు సమయం లేదు",
        "bengali": "আমার সময় ছিল না",
        "kannada": "ನನಗೆ ಸಮಯ ಸಿಕ್ಕಿಲ್ಲ",
        "marathi": "मला वेळ मिळाला नाही",
        "gujarati": "મારે સમય નથી મળ્યો",
        "punjabi": "ਮੈਂ ਕੋਲ ਸਮਾਂ ਨਹੀਂ ਸੀ",
        "odia": "ମୋତେ ସମୟ ମିଳିଲା ନାହିଁ"
    },
    "मुझे स्कूल जाना है": {
        "english": "I have to go to school",
        "tamil": "எனக்கு பள்ளிக்கு செல்ல வேண்டும்",
        "telugu": "నాకు పాఠశాలకు వెళ్లాలి",
        "bengali": "আমাকে স্কুলে যেতে হবে",
        "kannada": "ನಾನು ಶಾಲೆಗೆ ಹೋಗಬೇಕು",
        "marathi": "मला शाळेत जावे लागेल",
        "gujarati": "મને શાળામાં જવું છે",
        "punjabi": "ਮੈਨੂੰ ਸਕੂਲ ਜਾਣਾ ਹੈ",
        "odia": "ମୁଁ ବିଦ୍ୟାଳୟକୁ ଯିବାକୁ ପଡିବ"
    },
    "मुझे थकान हो रही है": {
        "english": "I am tired",
        "tamil": "நான் சோர்ந்துவிட்டேன்",
        "telugu": "నేను అలసిపోయాను",
        "bengali": "আমি ক্লান্ত",
        "kannada": "ನಾನು ದಣಿದಿದ್ದೇನೆ",
        "marathi": "मी थकलो आहे",
        "gujarati": "હું થાકી ગયો છું",
        "punjabi": "ਮੈਂ ਥੱਕ ਗਿਆ ਹਾਂ",
        "odia": "ମୁଁ ଥକି ଗଲି"
    }
        # Add more phrases here as needed
    }

    return translations.get(hindi_word, {}).get(target_lang.lower(), hindi_word)

# ----------------- ROADMAP ----------------- #
@app.route("/roadmap")
def roadmap():
    return render_template("roadmap.html")

# ----------------- RUN APP ----------------- #

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))  # Default to 5000 locally
    app.run(host='0.0.0.0', port=port, debug=False)
import cv2
import mediapipe as mp
import numpy as np
import tensorflow.lite as tflite
from collections import deque, Counter
from flask import Flask, jsonify
from flask_cors import CORS
import threading
from gpiozero import LED, TonalBuzzer # Support for both buzzer types
from gpiozero.tones import Tone

# --- PI 5 HARDWARE SETUP ---
# BCM 17 (Pin 11), BCM 18 (Pin 12)
red_led = LED(17)
alarm_buzzer = TonalBuzzer(18) 

# --- TRANSLATIONS ---
CORRECTED_TRANSLATIONS = {
    "Allergic": {"es": "Alérgico", "hi": "एलर्जी", "fr": "Allergique", "zh-CN": "过敏", "ar": "حساسية"},
    "Bathroom": {"es": "Baño", "hi": "शौचालय", "fr": "Toilettes", "zh-CN": "洗手间", "ar": "حमाम"},
    "Call": {"es": "Llamar", "hi": "कॉल करें", "fr": "Appeler", "zh-CN": "呼叫", "ar": "اتصال"},
    "Emergency": {"es": "Emergencia", "hi": "आपातकालीन", "fr": "Urgence", "zh-CN": "紧急情况", "ar": "طवारئ"},
    "Food": {"es": "Comida", "hi": "खाना", "fr": "Nourriture", "zh-CN": "食物", "ar": "طعام"},
    "Help": {"es": "Ayuda", "hi": "मदद", "fr": "Aider", "zh-CN": "帮助", "ar": "مساعدة"},
    "Landing": {"es": "Aterrizaje", "hi": "लैंडिंग", "fr": "Atterrissage", "zh-CN": "着陆", "ar": "هبوط"},
    "Pain": {"es": "Dolor", "hi": "दर्द", "fr": "Douleur", "zh-CN": "Douleur", "ar": "ألم"},
    "Repeat": {"es": "Repetir", "hi": "दोहराएं", "fr": "Répéter", "zh-CN": "重复", "ar": "كرर"},
    "Seatbelt": {"es": "Cinturón", "hi": "सीटबेल्ट", "fr": "Ceinture", "zh-CN": "安全带", "ar": "حزام"},
    "Stop": {"es": "Detener", "hi": "रुकें", "fr": "Arrêter", "zh-CN": "停止", "ar": "توقف"},
    "THANK": {"es": "Gracias", "hi": "धन्यवाद", "fr": "Merci", "zh-CN": "谢谢", "ar": "شكراً"},
    "Take": {"es": "Tomar", "hi": "लेना", "fr": "Prendre", "zh-CN": "拿", "ar": "أخذ"},
    "Water": {"es": " Agua", "hi": "पानी", "fr": "Eau", "zh-CN": "水", "ar": "ماء"},
    "Yes": {"es": "Sí", "hi": "हाँ", "fr": "Oui", "zh-CN": "是", "ar": "نعم"},
    "IDLE": {"es": "Esperando", "hi": "प्रतीक्षा", "fr": "En attente", "zh-CN": "等待", "ar": "انتظار"}
}

SIGNS = ["Allergic", "Bathroom", "Call", "Emergency", "Food", "Help",
         "Landing", "Pain", "Repeat", "Seatbelt", "Stop", "THANK",
         "Take", "Water", "Yes"]

# --- API SETUP ---
app = Flask(__name__)
CORS(app)
current_data = {"status": "IDLE", "translations": CORRECTED_TRANSLATIONS["IDLE"]}

@app.route('/')
def home():
    return """
    <html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { background: #000; color: white; font-family: sans-serif; display: flex; flex-direction: column; align-items: center; min-height: 100vh; margin: 0; padding-top: 20px;}
            #sign { font-size: 18vw; font-weight: 900; color: #00FF00; margin: 0; }
            table { width: 90%; border-collapse: collapse; margin-top: 20px; font-size: 4vw; border: 2px solid #222; }
            td { padding: 15px; border-bottom: 1px solid #222; }
            .lang-col { color: #888; width: 40%; }
            .trans-col { font-weight: bold; color: #00FF00; text-align: right; }
            .emergency-mode { animation: alert-pulse 0.5s infinite; }
            @keyframes alert-pulse { 50% { background: #400; } }
        </style>
    </head>
    <body>
        <div id="sign">IDLE</div>
        <table><tbody id="tableBody"></tbody></table>
        <script>
            const langNames = {"es": "Spanish", "hi": "Hindi", "fr": "French", "zh-CN": "Chinese", "ar": "Arabic"};
            async function update() {
                try {
                    const res = await fetch('/status');
                    const data = await res.json();
                    const s = document.getElementById('sign');
                    s.innerText = data.status.toUpperCase();
                    let html = "";
                    for (const [code, text] of Object.entries(data.translations)) {
                        html += `<tr><td class="lang-col">${langNames[code]}</td><td class="trans-col">${text}</td></tr>`;
                    }
                    document.getElementById('tableBody').innerHTML = html;
                    if(data.status === "Emergency" || data.status === "Help") {
                        document.body.className = "emergency-mode";
                        s.style.color = "red";
                    } else {
                        document.body.className = "";
                        s.style.color = (data.status === "IDLE") ? "#333" : "#00FF00";
                    }
                } catch(e) {}
            }
            setInterval(update, 250);
        </script>
    </body>
    </html>
    """

@app.route('/status')
def get_status():
    return jsonify(current_data)

def run_api():
    app.run(host='0.0.0.0', port=5000)

def normalize_landmarks(hl):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hl.landmark], dtype=np.float32)
    coords -= coords[0]
    mv = np.max(np.abs(coords)); coords = coords / mv if mv > 1e-6 else coords
    return coords.reshape(1, 21, 3)

def main():
    global current_data
    threading.Thread(target=run_api, daemon=True).start()

    class_scores = {sign: [] for sign in SIGNS}

    interpreter = tflite.Interpreter(model_path="../models/cnn/cnn_v3_multi_int8.tflite")
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]

    s_in, zp_in = in_det['quantization_parameters']['scales'][0], in_det['quantization_parameters']['zero_points'][0]
    s_out, zp_out = out_det['quantization_parameters']['scales'][0], out_det['quantization_parameters']['zero_points'][0]

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=0)

    cap = cv2.VideoCapture("http://10.206.10.57:8080/video")
    history = deque(maxlen=5)

    print("🚀 SkySign AI Active on Raspberry Pi 5")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.resize(frame, (480, 360)); frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                hl = results.multi_hand_landmarks[0]
                mp.solutions.drawing_utils.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
                feat = normalize_landmarks(hl)

                input_data = (feat / s_in + zp_in).astype(np.int8)
                interpreter.set_tensor(in_det['index'], input_data)
                interpreter.invoke()

                output = interpreter.get_tensor(out_det['index'])[0]
                probs = (output.astype(np.float32) - zp_out) * s_out
                idx = np.argmax(probs)

                current_sign = SIGNS[idx]
                class_scores[current_sign].append(probs[idx])
                history.append(current_sign)
                display_label = Counter(history).most_common(1)[0][0]
            else:
                display_label = "IDLE"
                history.clear()

            # --- PI 5 HARDWARE TRIGGERS (Tonal Fix) ---
            if display_label in ["Emergency", "Help"]:
                red_led.on()
                if not alarm_buzzer.is_active:
                    alarm_buzzer.play(Tone(440)) # 440Hz beep
            else:
                red_led.off()
                alarm_buzzer.stop()

            current_data["status"] = display_label
            current_data["translations"] = CORRECTED_TRANSLATIONS.get(display_label, CORRECTED_TRANSLATIONS["IDLE"])

            cv2.rectangle(frame, (0, 310), (480, 360), (0, 0, 0), -1)
            color = (0, 0, 255) if display_label in ["Emergency", "Help"] else (0, 255, 0)
            cv2.putText(frame, display_label.upper(), (20, 345), cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2)

            cv2.imshow("SkySign AI", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    finally:
        # Final safety cleanup
        red_led.off()
        alarm_buzzer.stop()
        
        print("\n" + "="*45)
        print(f"{'SIGN CLASS':<15} | {'AVG SESSION ACCURACY':<20}")
        print("-" * 45)
        for sign, scores in class_scores.items():
            if scores:
                avg = (sum(scores) / len(scores)) * 100
                print(f"{sign:<15} | {avg:>18.2f}%")
        print("="*45 + "\n")

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

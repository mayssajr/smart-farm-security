from flask import Flask, render_template_string, Response, jsonify
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import spidev
import time
import face_recognition
import os
import RPi.GPIO as GPIO


BUZZER_PIN = 18
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT, initial=GPIO.LOW)

def buzzer_alert(duration=0.25):
    GPIO.output(BUZZER_PIN, GPIO.HIGH)
    time.sleep(duration)
    GPIO.output(BUZZER_PIN, GPIO.LOW)


spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 1000000

def read_mcp3002(channel=0):
    command = [0b11000000 | (channel << 5), 0]
    resp = spi.xfer2(command)
    value = ((resp[0] & 0x03) << 8) | resp[1]
    return value

def mq135_ppm():
    raw = read_mcp3002(0)
    voltage = (raw / 1023.0) * 5.0
    ppm = voltage * 100
    return ppm

# ============================================================
# TFLITE ANIMAL CLASSIFIER
# ============================================================
interpreter = tflite.Interpreter(model_path="animal_classifier.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ['Poule', 'bear_png', 'chat', 'cheval', 'chien',
               'elephant', 'fox', 'lion', 'mouton', 'pig', 'tiger']

wild_classes = ["lion", "tiger", "pig", "fox", "elephant", "bear_png"]
domestic_classes = ["chien", "chat", "Poule", "mouton", "cheval"]

IMG = 224
weight_factor = 5.0

KNOWN_FACES_DIR = "known_faces"
known_encodings = []
known_names = []

if os.path.exists(KNOWN_FACES_DIR):
    for filename in os.listdir(KNOWN_FACES_DIR):
        path = os.path.join(KNOWN_FACES_DIR, filename)
        img = face_recognition.load_image_file(path)
        enc = face_recognition.face_encodings(img)[0]
        known_encodings.append(enc)
        known_names.append(os.path.splitext(filename)[0])

# ============================================================
# activation camera
# ============================================================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erreur : caméra introuvable.")
    exit()

# ============================================================
# interface
# ============================================================
last_detection_info = {
    "air": 0,
    "last": "Aucune",
    "state": "domestic"
}

# ============================================================
# VIDEO STREAM
# ============================================================
def generate_frames():
    global last_detection_info

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        air_ppm = mq135_ppm()

        # ====================== detection animaux======================
        img = cv2.resize(frame, (IMG, IMG))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = img_rgb.astype("float32") / 255.0
        img_rgb = np.expand_dims(img_rgb, axis=0)

        interpreter.set_tensor(input_details[0]['index'], img_rgb)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]['index'])[0]

        preds_adj = preds.copy()
        for i, a in enumerate(class_names):
            if a in wild_classes:
                preds_adj[i] *= weight_factor
        preds_adj /= np.sum(preds_adj)

        idx = np.argmax(preds_adj)
        confidence = preds_adj[idx]
        animal = class_names[idx]

        is_wild = animal in wild_classes
        status = "Wild" if is_wild else "Domestic"

        if is_wild:
            buzzer_alert(0.3)

        label_animal = f"{animal} ({status}) {confidence*100:.1f}%"

        cv2.putText(frame, label_animal, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0,0,255) if is_wild else (0,255,0), 2)

        cv2.putText(frame, f"Air: {air_ppm:.1f} ppm", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        # ====================== reconnaissance faciale ======================
        small = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
        rgb_small = small[:, :, ::-1]

        faces = face_recognition.face_locations(rgb_small)
        encs = face_recognition.face_encodings(rgb_small, faces)

        for (top, right, bottom, left), enc in zip(faces, encs):
            matches = face_recognition.compare_faces(known_encodings, enc)
            name = "Inconnu"

            dist = face_recognition.face_distance(known_encodings, enc)
            best = np.argmin(dist) if len(dist) > 0 else None

            if best is not None and matches[best]:
                name = known_names[best]

            if name == "Inconnu":
                buzzer_alert(0.25)

            top *= 4; right *= 4; bottom *= 4; left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
            cv2.putText(frame, name, (left, bottom+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        last_detection_info["air"] = round(air_ppm, 1)
        last_detection_info["last"] = label_animal
        last_detection_info["state"] = "wild" if is_wild else "domestic"

        ret2, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               jpeg.tobytes() +
               b'\r\n')

# ============================================================
# interface flask
# ============================================================
app = Flask(__name__)

HTML_PAGE = """
<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>Smart Farm Security</title>
<style>

    body {
        margin: 0;
        padding: 0;
        background: #0d1117;
        color: #ffffff;
        font-family: 'Segoe UI', Tahoma, sans-serif;
        display: flex;
        flex-direction: column;
        align-items: center;
    }

    header {
        width: 100%;
        padding: 20px 0;
        background: #161b22;
        box-shadow: 0 0 10px rgba(0,0,0,0.6);
        text-align: center;
        font-size: 26px;
        font-weight: bold;
        letter-spacing: 1px;
    }

    .container {
        margin-top: 30px;
        max-width: 900px;
        width: 95%;
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 25px;
    }

    .video-card {
        background: #161b22;
        padding: 15px;
        border-radius: 14px;
        box-shadow: 0 0 15px rgba(0,0,0,0.5);
        text-align: center;
        width: 100%;
    }

    .video-card h2 {
        margin-bottom: 15px;
        font-size: 22px;
        color: #58a6ff;
        font-weight: 600;
    }

    .video-box {
        border-radius: 12px;
        overflow: hidden;
        border: 2px solid #30363d;
    }

    img {
        width: 100%;
        max-height: 480px;
        display: block;
    }

    .status-card {
        background: #161b22;
        padding: 15px;
        border-radius: 14px;
        width: 100%;
        box-shadow: 0 0 15px rgba(0,0,0,0.5);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .status-text {
        font-size: 20px;
    }

    .led {
        width: 22px;
        height: 22px;
        border-radius: 50%;
        background-color: #2ea043;
        box-shadow: 0 0 10px #2ea043;
        transition: 0.3s ease;
    }

    .led.wild {
        background-color: #ff4d4d;
        box-shadow: 0 0 12px #ff4d4d;
    }

    footer {
        margin-top: 35px;
        padding: 10px;
        color: #555;
        font-size: 14px;
    }

</style>

<script>
setInterval(() => {
    fetch('/data')
    .then(r => r.json())
    .then(d => {
        document.getElementById("air_value").innerHTML = d.air + " ppm";
        document.getElementById("last_detection").innerHTML = d.last;

        const led = document.getElementById("status_led");
        if (d.state === "wild") {
            led.classList.add("wild");
        } else {
            led.classList.remove("wild");
        }
    });
}, 1000);
</script>

</head>

<body>

<header>Smart Farm Security Dashboard</header>

<div class="container">

    <div class="video-card">
        <h2>Flux Vidéo en Temps Réel</h2>
        <div class="video-box">
            <img src="/video">
        </div>
    </div>

    <div class="status-card">
        <div class="status-text">
            Qualité de l'air : <span id="air_value">-- ppm</span><br>
            Dernière détection : <span id="last_detection">--</span>
        </div>
        <div id="status_led" class="led"></div>
    </div>

</div>

<footer>Smart Farm AI & IoT Monitoring – Raspberry Pi</footer>

</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

@app.route("/video")
def video():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/data")
def data():
    return jsonify(last_detection_info)


if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, debug=False)
    finally:
        cap.release()
        spi.close()
        GPIO.cleanup()

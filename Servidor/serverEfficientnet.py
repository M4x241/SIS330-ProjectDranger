from flask import Flask, request, jsonify
import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
import os
from datetime import datetime
from torchvision import transforms

app = Flask(__name__)

# =========================
# CONFIG
# =========================
MODEL_PATH = "/home/ghost/Documents/Proyectos/python/Proyecto Python/Servidor/efficientnet_results/efficientnet_smoke.pth"
SAVE_DIR = "/home/ghost/Documents/Proyectos/python/Proyecto Python/Servidor/signalSmoke"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# MODELO EfficientNet
# =========================
model = timm.create_model(
    "efficientnet_b0",
    pretrained=False,
    num_classes=2      # ✅ SOLO aquí se define
)

# cargar pesos
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)

model.to(DEVICE)
model.eval()

# =========================
# TRANSFORMACIONES
# (DEBEN SER IDENTICAS AL TRAIN)
# =========================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ✅ Para clasificación multiclase
softmax = nn.Softmax(dim=1)

# =========================
# ENDPOINT
# =========================
@app.route('/upload', methods=['POST'])
def upload():

    # validar request
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file_bytes = request.files['image'].read()

    # bytes → OpenCV
    npimg = np.frombuffer(file_bytes, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "Invalid image"}), 400

    # BGR → RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # preprocess
    input_tensor = transform(rgb).unsqueeze(0).to(DEVICE)

    # =========================
    # INFERENCIA
    # =========================
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = softmax(outputs)[0]

        prob_background = probs[0].item()
        prob_smoke = probs[1].item()

    print(f"Background: {prob_background:.3f} | Smoke: {prob_smoke:.3f}")

    # 🔥 umbral configurable
    detected = prob_smoke > 0.7

    # =========================
    # GUARDAR SI DETECTA HUMO
    # =========================
    saved_path = None

    if detected:
        filename = datetime.now().strftime("smoke_%Y%m%d_%H%M%S_%f.jpg")
        saved_path = os.path.join(SAVE_DIR, filename)

        cv2.imwrite(saved_path, frame)
        print(f"🔥 Humo detectado → {saved_path}")

    # =========================
    # RESPUESTA
    # =========================
    return jsonify({
        "status": "success",
        "probabilities": {
            "background": prob_background,
            "smoke": prob_smoke
        },
        "detected": detected,
        "saved_path": saved_path
    }), 200


# =========================
# RUN SERVER
# =========================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5500, debug=False)
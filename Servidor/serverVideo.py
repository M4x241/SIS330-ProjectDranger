from flask import Flask, request
import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime

app = Flask(__name__)

# Modelo entrenado
model = YOLO('/home/ghost/Documents/Proyectos/python/Proyecto Python/Servidor/best100Yolo.pt')
save_dir = "/home/ghost/Documents/Proyectos/python/Proyecto Python/Servidor/signalSmoke"

# crear carpeta si no existe
os.makedirs(save_dir, exist_ok=True)


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image'].read()

    # bytes → imagen OpenCV
    npimg = np.frombuffer(file, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Inferencia YOLO
    results = model(frame, conf=0.5)

    result = results[0]
    detections = len(result.boxes)

    # ✅ Si detecta humo → guardar imagen
    if detections > 0:
        # nombre único usando timestamp
        filename = datetime.now().strftime("smoke_%Y%m%d_%H%M%S_%f.jpg")
        filepath = os.path.join(save_dir, filename)

        cv2.imwrite(filepath, frame)

        print(f"🔥 Humo detectado → imagen guardada en: {filepath}")

    return {
        "status": "success",
        "detections": detections
    }, 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5500)
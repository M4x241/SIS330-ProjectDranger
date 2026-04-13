from ultralytics import YOLO
import os

# 1. Definir rutas absolutas para evitar problemas en Linux
# Usamos la ruta exacta donde estás parado
DATASET_DIR = "/home/ghost/Downloads/Deteccion de Fuego y Humo.v1i.yolov8"
YAML_PATH = os.path.join(DATASET_DIR, "data.yaml")

def entrenar_modelo():
    print(f"--- Iniciando entrenamiento usando dataset en: {DATASET_DIR} ---")
    
    # 2. Cargar el modelo base pre-entrenado (Transfer Learning)
    # Usamos 'n' (nano) para que sea ultrarrápido en Unity Sentis luego.
    model = YOLO('yolov8n.pt') 

    # 3. Entrenar el modelo
    # data: Ruta al archivo .yaml
    # epochs: Cuántas veces el modelo verá todo el dataset (50 es buen inicio)
    # imgsz: Tamaño de imagen (640 es estándar y equilibrado)
    # device: Forzamos el uso de tu GPU CUDA
    results = model.train(
        data=YAML_PATH, 
        epochs=50, 
        imgsz=640, 
        device='cuda:0',
        workers=4 # Ajusta según tus núcleos de CPU para cargar imágenes rápido
    )
    
    print("--- Entrenamiento finalizado ---")
    print(f"Los pesos entrenados están en: {results.save_dir}")

if __name__ == '__main__':
    entrenar_modelo()
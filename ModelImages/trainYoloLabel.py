from ultralytics import YOLO
import os

# 1. Rutas del sistema (KDE Neon / Ghost)
DATASET_DIR = "/home/ghost/Music/humo"
YAML_PATH = os.path.join(DATASET_DIR, "data.yaml")
    
def entrenar_modelo():
    print(f"--- Iniciando entrenamiento en: {DATASET_DIR} ---")
    
    # 2. Cargar modelo base
    model = YOLO('yolov8n.pt') 

    # 3. Entrenar el modelo
    # name='labelSmoke': Esto creará la carpeta 'labelSmoke' dentro de runs/detect/
    results = model.train(
        data=YAML_PATH, 
        epochs=100,        # Subimos a 100 para dar más tiempo de aprendizaje con pocas fotos
        imgsz=640, 
        device='cuda:0',    
        workers=4,
        name='labelSmoke', # Nombre del experimento y de los pesos
        
        mosaic=1.0,        
        mixup=0.1,          
        degrees=10.0,      # Rotaciones leves para el ángulo del dron
        perspective=0.0005 # Simula cambios de perspectiva de vuelo
    )
    
    print("--- Entrenamiento finalizado ---")
    # El archivo se guardará en runs/detect/labelSmoke/weights/best.pt
    print(f"Tus pesos 'labelSmoke' están en: {results.save_dir}/weights/bestlabelSmoke.pt")

if __name__ == '__main__':
    entrenar_modelo()cd 
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import os

# --- CONFIGURACIÓN ---
# 1. Ruta absoluta a tus pesos entrenados ('best.pt')
# Asegúrate de que esta ruta coincida con donde terminó tu entrenamiento.
# Si hiciste varios entrenamientos, podría ser 'train2', 'train3', etc.
PATH_PESOS_ENTRENADOS = '/home/ghost/Documents/Proyectos/python/Proyecto Python/Servidor/best100Yolo.pt'

# 2. Ruta a la imagen de tu entorno Unity que quieres probar
# Puedes usar una captura de pantalla fresca de Dranger/Martijn.
PATH_IMAGEN_PRUEBA = '/home/ghost/Documents/Proyectos/python/Proyecto Python/Servidor/Screenshot_20260318_115043.png'
# ---------------------

def probar_deteccion():
    # Verificar que existen los pesos
    if not os.path.exists(PATH_PESOS_ENTRENADOS):
        print(f"ERROR: No se encontraron los pesos en {PATH_PESOS_ENTRENADOS}")
        print("Revisa la ruta donde se guardó tu entrenamiento.")
        return

    # 1. Cargar TU modelo entrenado
    print(f"Cargando modelo especialista desde: {PATH_PESOS_ENTRENADOS}")
    model = YOLO(PATH_PESOS_ENTRENADOS)

    # Solo para verificar qué aprendió el modelo:
    print(f"Clases que el modelo puede detectar: {model.names}")

    # 2. Ejecutar la inferencia (predicción)
    # conf=0.5: Subimos un poco el umbral de confianza para evitar falsos positivos
    print(f"Procesando imagen: {PATH_IMAGEN_PRUEBA}")
    results = model(PATH_IMAGEN_PRUEBA, conf=0.5, device='cuda:0') # Usamos CUDA para velocidad

    # 3. Visualizar los resultados
    for r in results:
        # Dibujar las cajas y etiquetas sobre la imagen
        im_array = r.plot()
        
        # Guardar el resultado en disco (útil si la ventana OpenCV falla en Wayland)
        path_resultado = "resultado_deteccion.jpg"
        cv2.imwrite(path_resultado, im_array)
        print(f"Imagen de resultado guardada como: {path_resultado}")

        img_rgb = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)

        plt.imshow(img_rgb)
        plt.title("Prueba de Modelo de Humo Especializado")
        plt.axis("off")
        plt.show()
if __name__ == '__main__':
    # Solución rápida para el error de Wayland que vimos antes en tu Linux
    os.environ["QT_QPA_PLATFORM"] = "xcb" 
    probar_deteccion()
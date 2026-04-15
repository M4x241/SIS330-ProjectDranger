# DRANGER: Vigilancia Autónoma Forestal con IA Multimodal

Este proyecto implementa un sistema de centinela autónomo de última generación que fusiona **Deep Reinforcement Learning (DRL)** para la navegación inteligente con modelos avanzados de **Computer Vision** y **Audio Analysis**. El objetivo principal es la detección ultra-temprana de incendios forestales y actividades humanas ilícitas, como la tala mediante motosierras, en entornos bioma-críticos.

## 🚀 1. Preparación del Entorno

Para garantizar la estabilidad de los modelos y la comunicación con el simulador, se requiere un entorno controlado. Se recomienda el uso de **Python 3.10+** y hardware con soporte **CUDA** (controladores NVIDIA) para acelerar la inferencia en tiempo real.

1.  **Clonar el repositorio:**
    Obtén la versión más reciente del código fuente:
    ```bash
    git clone <url-del-repositorio>
    cd SIS330-ProjectDranger
    ```

2.  **Crear y activar un entorno virtual:**
    Esto evita conflictos entre dependencias globales:
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    ```

3.  **Instalar dependencias:**
    El archivo `requirements.txt` incluye librerías críticas como `ultralytics` (YOLO), `torch` (PyTorch), `librosa` (procesamiento de audio) y `opencv-python`:
    ```bash
    pip install -r requirements.txt
    ```

---

## 📸 2. Pipeline de Visión Artificial (Humo e Incendios)

El sistema de visión utiliza una arquitectura híbrida para minimizar errores. Sigue este flujo estrictamente para garantizar la calidad del entrenamiento:

### A. Procesamiento del Dataset
La calidad de la detección depende de cómo se preparan los datos en `ModelImages/ProcessinData/`:

1.  **Generar Fondos (`CrearBackgroudng.py`):** Este script es vital para reducir los **falsos positivos**. Procesa imágenes del bosque sin humo para que el modelo aprenda a ignorar elementos naturales que podrían confundirse con columnas de fuego (como niebla o nubes bajas).
2.  **Preparar Dataset (`prepararDataset.py`):** Una vez generadas las anotaciones y los fondos, este script organiza la arquitectura de carpetas, aplica etiquetas y realiza el **split de datos** (típicamente 80/20) entre entrenamiento y validación, asegurando que el modelo sea evaluado con datos que nunca ha visto.

### B. Entrenamiento
Con el dataset estructurado en la carpeta raíz de imágenes, procede a la fase de aprendizaje:

* **Entrenamiento YOLO:** Ejecuta `trainyoloLabel.py`. Este script configura los hiperparámetros de la red **YOLOv8** para la detección de Bounding Boxes. Al finalizar, generará un archivo `best.pt` que se utilizará para las pruebas y la ejecución en vivo.

---

## 🔊 3. Pipeline de Audio (Detección de Tala)

La detección acústica permite al dron "escuchar" amenazas que están fuera de su línea de visión (debajo del follaje o en condiciones de mucho humo).

1.  Dirígete a `ModelSounds/`.
2.  Abre y ejecuta el notebook `ChainsawTrained.ipynb`:
    * **Gestión de Datos:** Si no cuentas con los archivos `.wav`, el notebook incluye una celda de descarga automática para obtener el dataset de sonidos ambientales y maquinaria.
    * **Procesamiento:** El audio se transforma en **Espectrogramas Mel**, una representación visual de la frecuencia que permite a la red neuronal clasificar sonidos de motosierras con alta precisión frente al ruido del viento o motores del dron.

---

## 🖥️ 4. Ejecución del Sistema y Servidores

Para que la simulación en Unity interactúe con el cerebro de IA en Python, debes levantar la infraestructura de servidores en `Servidor/`:

* **`serverUnity.py`:** Actúa como el puente principal de datos. Recibe la telemetría del dron y envía las órdenes de control generadas por el modelo de navegación.
* **`serverVideo.py`:** Transmite el flujo visual del dron hacia una interfaz de monitoreo, permitiendo la supervisión humana remota.
* **Carpeta `signalSmoke/`:** Este es el **repositorio de evidencia**. Cada vez que la IA confirma una detección, guarda una captura de pantalla con marca de tiempo aquí para su posterior análisis forense.

---

## 📁 5. Estructura de Carpetas Detallada

* **`ModelImages/`**: Núcleo del entrenamiento visual. Incluye la lógica de procesamiento de datos y almacenamiento de pesos del modelo.
* **`ModelSounds/`**: Entorno de IA acústica. Contiene los modelos `.pth` optimizados para la detección de frecuencias industriales.
* **`Servidor/`**: Capa de comunicación. Es el software que permite que el dron "viva" fuera del código y opere en un entorno simulado o real.
* **`ModelImages/Test/`**: Suite de validación. Incluye scripts como `TestSmokeYololabel.py` para realizar pruebas de estrés con imágenes nuevas y verificar la precisión del modelo entrenado.
* **`DroneEscape.onnx`**: El archivo de exportación del modelo **DRL**. Contiene la política de vuelo entrenada en Unity para evitar obstáculos y optimizar rutas de patrullaje.

---

## 🛠️ 6. Pruebas de Resultados

Para validar que la instalación fue exitosa y los modelos funcionan según lo esperado, ejecuta:

* **Visión:** `python ModelImages/Test/TestSmokeYololabel.py` (debería abrir una ventana mostrando las cajas de detección sobre imágenes de prueba).
* **Audio:** `python ModelSounds/Test/testAudio.py` (analizará un clip de audio de muestra y devolverá la probabilidad de que sea una motosierra).

**Desarrollado por:** Max Jherzon Rodas Palacios
**Proyecto:** SIS330 - Inteligencia Artificial II (USFX)
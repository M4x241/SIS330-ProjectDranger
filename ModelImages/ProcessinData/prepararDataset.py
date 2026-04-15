import json
import os
import glob
import random
import shutil

def split_dataset(dataset_path, train_ratio=0.75):
    # Obtener todos los archivos JSON en la carpeta raíz
    json_files = glob.glob(os.path.join(dataset_path, "*.json"))
    paired_files = [(json_file, os.path.splitext(json_file)[0] + ".jpg") for json_file in json_files]
    paired_files += [(json_file, os.path.splitext(json_file)[0] + ".png") for json_file in json_files]

    # Filtrar pares válidos (donde existan ambos archivos)
    valid_pairs = [(json_file, img_file) for json_file, img_file in paired_files if os.path.exists(img_file)]

    # Mezclar aleatoriamente los pares
    random.shuffle(valid_pairs)

    # Dividir en train y test
    train_size = int(len(valid_pairs) * train_ratio)
    train_pairs = valid_pairs[:train_size]
    test_pairs = valid_pairs[train_size:]

    # Crear carpetas train y test
    train_path = os.path.join(dataset_path, "train")
    
    # Procesar imágenes de fondo
    background_path = "/home/ghost/Music/background"
    background_images = glob.glob(os.path.join(background_path, "*.jpg")) + glob.glob(os.path.join(background_path, "*.png"))
    
    # Crear pares de fondo con archivos .txt vacíos
    background_pairs = [(img_file, os.path.splitext(img_file)[0] + ".txt") for img_file in background_images]
    for _, txt_file in background_pairs:
        with open(txt_file, 'w') as f:
            pass  # Crear archivo .txt vacío si no existe

    # Mezclar imágenes de fondo con el dataset
    random.shuffle(background_pairs)
    train_size_bg = int(len(background_pairs) * train_ratio)
    train_pairs += background_pairs[:train_size_bg]
    test_pairs += background_pairs[train_size_bg:]
    test_path = os.path.join(dataset_path, "test")
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    # Mover archivos a sus respectivas carpetas
    for json_file, img_file in train_pairs:
        shutil.move(json_file, os.path.join(train_path, os.path.basename(json_file)))
        shutil.move(img_file, os.path.join(train_path, os.path.basename(img_file)))

    for json_file, img_file in test_pairs:
        shutil.move(json_file, os.path.join(test_path, os.path.basename(json_file)))
        shutil.move(img_file, os.path.join(test_path, os.path.basename(img_file)))

    print("¡Datos divididos en train y test!")

def convert_labelme_to_yolo(dataset_path):
    # Carpetas a procesar
    subsets = ['train', 'test']
    
    for subset in subsets:
        subset_path = os.path.join(dataset_path, subset)
        if not os.path.exists(subset_path): continue

        # Crear carpetas internas
        img_dir = os.path.join(subset_path, 'images')
        lbl_dir = os.path.join(subset_path, 'labels')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        # Buscar todos los JSON
        json_files = glob.glob(os.path.join(subset_path, "*.json"))
        
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            img_name = data['imagePath']
            # Nombre del archivo txt
            txt_name = os.path.splitext(os.path.basename(json_file))[0] + ".txt"
            
            # Mover imagen a /images
            old_img_path = os.path.join(subset_path, img_name)
            if os.path.exists(old_img_path):
                os.rename(old_img_path, os.path.join(img_dir, img_name))

            # Convertir formas a YOLO
            yolo_lines = []
            img_h = data['imageHeight']
            img_w = data['imageWidth']

            for shape in data['shapes']:
                # Asumimos que la clase 0 es 'humo'
                points = shape['points']
                # Calcular bounding box
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                
                # Normalizar coordenadas
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                
                x_center = (x_min + x_max) / 2 / img_w
                y_center = (y_min + y_max) / 2 / img_h
                w = (x_max - x_min) / img_w
                h = (y_max - y_min) / img_h
                
                yolo_lines.append(f"0 {x_center} {y_center} {w} {h}")

            # Guardar el .txt en /labels
            with open(os.path.join(lbl_dir, txt_name), 'w') as f:
                f.write("\n".join(yolo_lines))
            
            # Eliminar el JSON original para limpiar
            os.remove(json_file)
            
    print("¡Dataset organizado y convertido!")

if __name__ == "__main__":
    dataset_path = "/home/ghost/Music/humo"
    split_dataset(dataset_path)
    convert_labelme_to_yolo(dataset_path)
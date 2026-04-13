import json
import os
import glob

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
    convert_labelme_to_yolo("/home/ghost/Music/humo")
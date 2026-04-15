import os

# 📂 carpeta donde están tus imágenes
images_dir = "/home/ghost/Music/background/"

# 📂 carpeta donde se guardarán los labels
labels_dir = "/home/ghost/Music/background"

# extensiones válidas
image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# crear carpeta labels si no existe
os.makedirs(labels_dir, exist_ok=True)

created = 0
skipped = 0

for filename in os.listdir(images_dir):
    if filename.lower().endswith(image_extensions):

        name_without_ext = os.path.splitext(filename)[0]
        label_path = os.path.join(labels_dir, name_without_ext + ".txt")

        # ⚠️ no sobrescribir labels existentes
        if os.path.exists(label_path):
            skipped += 1
            continue

        # crear archivo vacío
        open(label_path, "w").close()
        created += 1

print(f"✅ Labels vacíos creados: {created}")
print(f"⏭️ Labels existentes omitidos: {skipped}")
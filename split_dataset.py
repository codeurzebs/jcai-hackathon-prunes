import os
import shutil
import random

# ------------------- Configuration -------------------

RAW_DIR = "data/raw"              # Où se trouvent tes images d'origine
DEST_DIR = "data/processed"       # Où seront stockées les données divisées
TRAIN_RATIO = 0.8                 # Pourcentage d'entraînement

# -----------------------------------------------------

def split_images():
    # Créer les dossiers train/ et val/ si pas déjà là
    train_dir = os.path.join(DEST_DIR, "train")
    val_dir = os.path.join(DEST_DIR, "val")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Liste des classes (chaque sous-dossier = une classe)
    classes = [d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))]

    for cls in classes:
        print(f"📁 Traitement de la classe : {cls}")

        class_path = os.path.join(RAW_DIR, cls)
        images = os.listdir(class_path)
        random.shuffle(images)

        split_index = int(len(images) * TRAIN_RATIO)
        train_images = images[:split_index]
        val_images = images[split_index:]

        # Dossiers pour cette classe
        train_class_dir = os.path.join(train_dir, cls)
        val_class_dir = os.path.join(val_dir, cls)

        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)

        # Copier les images
        for img in train_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(train_class_dir, img)
            shutil.copyfile(src, dst)

        for img in val_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(val_class_dir, img)
            shutil.copyfile(src, dst)

    print("\n✅ Division terminée ! Les images sont prêtes dans data/processed/\n")

# Lancer le script
if __name__ == "__main__":
    split_images()

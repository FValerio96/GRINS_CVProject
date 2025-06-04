import os
import json
import numpy as np
from PIL import Image
from pathlib import Path

# Percorso all'immagine placeholder di riferimento
REFERENCE_IMAGE_PATH = "placeholder_img/image_142_143_heading_0.jpg"

# Directory base del dataset
BASE_DIR = "data/processed"

# Città presenti nel dataset
CITIES = ["Bari", "Shibuya", "Manhattan"]

# File di log
LOG_FILE = "log.txt"

def load_reference_image(path):
    ref_img = Image.open(path).convert("RGB")
    return np.array(ref_img)

def is_placeholder(image_path, reference_array, tolerance=5):
    try:
        img = Image.open(image_path).convert("RGB")
        img_arr = np.array(img)
        if img_arr.shape != reference_array.shape:
            return False
        diff = np.abs(img_arr.astype(int) - reference_array.astype(int))
        return np.max(diff) <= tolerance
    except:
        return False

def remove_file(path, log_entries):
    if os.path.exists(path):
        os.remove(path)
        log_entries.append(f"Rimosso: {path}")

def clean_dataset():
    # carico l'array del img placeholder
    reference_array = load_reference_image(REFERENCE_IMAGE_PATH)
    log_entries = []

    for city in CITIES:
        city_path = os.path.join(BASE_DIR, city)
        embedding_path = os.path.join(city_path, f"{city.lower()}_clip.npy")
        removed_prefixes = set()

        print(f"\n🔍 Scansione della città: {city}")
        for root, _, files in os.walk(city_path):
            for file in files:
                if file.endswith(".jpg"):
                    img_path = os.path.join(root, file)
                    if is_placeholder(img_path, reference_array):
                        print(f"🗑️  Rimozione del placeholder: {img_path}")
                        os.remove(img_path)
                        log_entries.append(f"Rimosso: {img_path}")

                        name_stem = Path(file).stem
                        removed_prefixes.add(f"{city}_{name_stem}")

                        label_path = os.path.join(city_path, "masked_output", "labels", f"{name_stem}.json")
                        mask_path = os.path.join(city_path, "masked_output", "masks", f"{name_stem}.npy")

                        remove_file(label_path, log_entries)
                        remove_file(mask_path, log_entries)

        # Rimuovi embedding per questa città
        if os.path.exists(embedding_path):
            emb = np.load(embedding_path, allow_pickle=True).item()
            new_emb = {
                k: v for k, v in emb.items()
                if not any(k.startswith(prefix) for prefix in removed_prefixes)
            }
            np.save(embedding_path, new_emb)
            log_entries.append(f"✅ Salvati {len(new_emb)} embedding puliti in {embedding_path}")
        else:
            log_entries.append(f"⚠️  File embedding non trovato: {embedding_path}")

    # Scrittura del log finale
    with open(LOG_FILE, "w") as log_file:
        for entry in log_entries:
            log_file.write(entry + "\n")

    print(f"\n📝 Log delle operazioni salvato in {LOG_FILE}")

if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent.parent.parent)
    clean_dataset()

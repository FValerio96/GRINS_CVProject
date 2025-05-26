import os
import json
from pathlib import Path

def fix_labels_for_city(city_dir: Path):
    labels_dir = city_dir / "masked_output" / "labels"
    if not labels_dir.exists():
        print(f"[WARNING] No labels found in: {labels_dir}")
        return

    for file in labels_dir.iterdir():
        if file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            try:
                with open(file, "r", encoding="utf-8") as f:
                    content = json.load(f)  # Prova a leggere come JSON
                # Se ci riesce, rinomina il file
                new_path = file.with_suffix(".json")
                file.rename(new_path)
                print(f"[OK] Rinomina: {file.name} → {new_path.name}")
            except Exception as e:
                print(f"[SKIP] {file.name} non è un JSON valido: {e}")

def fix_all_cities(root_dir="data/processed"):
    cities = ["Bari", "Manhattan", "Shibuya"]
    for city in cities:
        city_path = Path(root_dir) / city
        print(f"\n🔍 Processing city: {city}")
        fix_labels_for_city(city_path)

if __name__ == "__main__":
    fix_all_cities()

import os
from pathlib import Path

def fix_masks_for_city(city_dir: Path):
    masks_dir = city_dir / "masked_output" / "masks"
    if not masks_dir.exists():
        print(f"[WARNING] Cartella maschere non trovata in: {masks_dir}")
        return

    for file in masks_dir.iterdir():
        if file.name.endswith(".jpg.npy"):
            new_name = file.name.replace(".jpg.npy", ".npy")
            new_path = file.with_name(new_name)
            file.rename(new_path)
            print(f"[OK] Rinomina: {file.name} → {new_name}")

def fix_all_cities(root_dir="data/processed"):
    cities = ["Manhattan", "Shibuya"]
    for city in cities:
        city_path = Path(root_dir) / city
        print(f"\n🔍 Processing city: {city}")
        fix_masks_for_city(city_path)

if __name__ == "__main__":
    fix_all_cities()

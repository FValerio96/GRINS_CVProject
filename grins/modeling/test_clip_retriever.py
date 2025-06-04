import os
import json
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from grins.modeling.clip_retriever import CLIPRetriever

def visualize_top_matches(json_path, base_dir="data/processed"):
    with open(json_path, "r") as f:
        top_matches = json.load(f)

    for match in top_matches:
        name1 = match["name1"]  # Bari
        name2 = match["name2"]  # Shibuya

        city1, image1, heading1, region1 = parse_info(name1)
        city2, image2, heading2, region2 = parse_info(name2)

        retriever1 = CLIPRetriever(base_dir, city1, "dummy.npy")
        retriever2 = CLIPRetriever(base_dir, city2, "dummy.npy")

        crop1, full1 = get_crop_and_full(retriever1, base_dir, city1, image1, heading1, region1)
        crop2, full2 = get_crop_and_full(retriever2, base_dir, city2, image2, heading2, region2)

        show_comparison(full1, crop1, full2, crop2, name1, name2)

def parse_info(name):
    parts = name.split("_")
    city = parts[0]
    image_id = "_".join(parts[1:4])  # es. image_142_143
    heading = parts[5]               # es. 0
    region = int(parts[7])           # es. region_0 → 0
    return city, image_id, heading, region

def get_crop_and_full(retriever, base_dir, city, image_id, heading, region_idx):
    label_path = os.path.join(base_dir, city, "masked_output", "labels", f"{image_id}_heading_{heading}.json")
    data, mask = retriever._load_json_and_mask(os.path.basename(label_path))
    image_path = os.path.join(base_dir, city, heading, f"{image_id}_heading_{heading}.jpg")
    image = Image.open(image_path).convert("RGB")
    region = data["regions"][region_idx]
    bbox = region["bbox"]
    class_id = region["class_id"]
    crop = retriever._get_region_crop(image, mask, bbox, class_id)
    return crop, image

def show_comparison(full1, crop1, full2, crop2, name1, name2):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    axs[0, 0].imshow(full1)
    axs[0, 0].set_title(f"Original: {name1}")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(full2)
    axs[0, 1].set_title(f"Original: {name2}")
    axs[0, 1].axis("off")

    axs[1, 0].imshow(crop1)
    axs[1, 0].set_title("Crop Bari")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(crop2)
    axs[1, 1].set_title("Crop Shibuya")
    axs[1, 1].axis("off")

    plt.tight_layout()
    plt.show()

# Esegui lo script
if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent.parent.parent)
    visualize_top_matches("retrieval_results/Bari_vs_Shibuya_top5_global.json")

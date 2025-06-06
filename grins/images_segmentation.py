from pathlib import Path
import pathlib
import typer
from loguru import logger
import os
import re
import ast
import torch
import numpy as np
import pandas as pd
from PIL import Image
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from tqdm import tqdm
from grins.config import PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR
import json

app = typer.Typer()

def merge_output_with_coordinates(output_file_path, coordinates_file_path, final_csv_path):
    """
    Merges output.csv and merged_coordinates.csv based on ID and sub_id, ensuring 
    only matching IDs are retained and aligned correctly.

    Parameters:
    output_file_path (str): Path to the output CSV file.
    coordinates_file_path (str): Path to the coordinates CSV file.
    final_csv_path (str): Path to save the final merged CSV.

    Returns:
    str: Path to the final saved CSV file.
    """
    # Load the files
    output_df = pd.read_csv(output_file_path)
    coordinates_df = pd.read_csv(coordinates_file_path)

    # Rename 'image_id' to 'ID' for consistency
    output_df.rename(columns={"image_id": "ID"}, inplace=True)

    # Remove IDs from coordinates_df that are not in output_df
    coordinates_df = coordinates_df[coordinates_df["ID"].isin(output_df["ID"])].copy()

    # Sort both DataFrames by ID and sub_id to match corresponding rows correctly
    output_df.sort_values(by=["ID", "sub_id"], inplace=True)
    coordinates_df.sort_values(by="ID", inplace=True)

    # Reset index after sorting
    output_df.reset_index(drop=True, inplace=True)
    coordinates_df.reset_index(drop=True, inplace=True)

    # Merge the data while ensuring correct alignment based on sorted order
    merged_df = output_df.copy()
    merged_df[["lon", "lat"]] = coordinates_df[["lon", "lat"]]

    # Save the final CSV
    merged_df.to_csv(final_csv_path, index=False)

    logger.info(f"Final CSV saved as: {final_csv_path}")

def extract_image_ids(image_path):
    filename = os.path.basename(image_path)
    numbers = re.findall(r'(\d+)', filename)
    if len(numbers) >= 2:
        return numbers[0], numbers[1]  # sub_id, group_id
    else:
        return numbers[0], None

class StreetViewDataset(Dataset):
    def __init__(self, image_paths, color_mapping, macro_mapping, processor):
        self.image_paths = image_paths
        self.color_mapping = color_mapping
        self.macro_mapping = macro_mapping
        self.processor = processor
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        return image_path, image, inputs

def custom_collate(batch):
    image_paths, images, inputs = zip(*batch)
    input_dict = {key: torch.cat([inp[key] for inp in inputs], dim=0) for key in inputs[0]}
    return image_paths, list(images), input_dict

def get_image_paths(base_path, angle_dirs=None):
    image_paths = []
    if angle_dirs is None:
        # Modalità flat (nessuna sottocartella)
        file_list = sorted([str(path) for path in Path(base_path).glob('*.png')])
        image_paths.extend(file_list)
    else:
        for angle in angle_dirs:
            path_list = Path(base_path) / angle
            file_list = sorted([str(path) for path in path_list.glob('*.jpg')])
            image_paths.extend(file_list)
    return image_paths


def process_batch(model, batch, color_mapping, macro_mapping, device, processor, output_folder):
    image_paths, images, inputs = batch
    inputs = {k: inputs[k].to(device) for k in inputs}

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_semantic_segmentation(
        outputs,
        target_sizes=[img.size[::-1] for img in images]
    )

    blended_images = []
    pixel_distributions = []

    for image_path, img, mask in zip(image_paths, images, results):
        mask = mask.cpu().numpy()
        output_image_np = np.array(img)

        # Colora segmenti secondo mapping
        for label in np.unique(mask):
            text_label = model.config.id2label.get(label, str(label))
            if text_label in color_mapping:
                output_image_np[mask == label] = color_mapping[text_label]

        # Salva immagine blended
        blended_image = Image.blend(img, Image.fromarray(output_image_np), alpha=0.7)
        blended_images.append(blended_image)

        # Salva maschera grezza (.npy)
        mask_save_dir = os.path.join(output_folder, "masks")
        os.makedirs(mask_save_dir, exist_ok=True)
        mask_filename = os.path.basename(image_path).replace(".jpg", "_mask.npy")
        np.save(os.path.join(mask_save_dir, mask_filename), mask)

        # Classi presenti
        class_ids = np.unique(mask)
        class_names = [model.config.id2label.get(cls, f"unknown_{cls}") for cls in class_ids]

        # Crea struttura JSON con info su classi e regioni
        labels_info = {
            "image": os.path.basename(image_path),
            "class_ids": class_ids.tolist(),
            "class_names": class_names,
            "regions": []
        }

        for cls_id in class_ids:
            mask_bin = (mask == cls_id).astype(np.uint8)
            ys, xs = np.where(mask_bin)
            if ys.size > 0 and xs.size > 0:
                x_min, x_max = int(xs.min()), int(xs.max())
                y_min, y_max = int(ys.min()), int(ys.max())
                labels_info["regions"].append({
                    "class_id": int(cls_id),
                    "class_name": model.config.id2label.get(cls_id, f"unknown_{cls_id}"),
                    "bbox": [x_min, y_min, x_max, y_max]
                })

        # Salva file .json
        labels_dir = os.path.join(output_folder, "labels")
        os.makedirs(labels_dir, exist_ok=True)
        json_filename = os.path.basename(image_path).replace(".png", "_labels.json")
        with open(os.path.join(labels_dir, json_filename), "w") as f:
            json.dump(labels_info, f, indent=2)

        # Calcola distribuzione pixel per macro-classi
        pixel_distribution = defaultdict(int)
        for label in np.unique(mask):
            text_label = model.config.id2label.get(label, f"unknown_{label}")
            pixel_count = np.sum(mask == label)
            pixel_distribution[macro_mapping.get(text_label, 'Unknown')] += pixel_count
        pixel_distributions.append(pixel_distribution)

    return image_paths, blended_images, pixel_distributions


def update_csv_data(image_paths, blended_images, pixel_distributions, output_folder, csv_data):
    for image_path, blended_image, pixel_distribution in zip(image_paths, blended_images, pixel_distributions):
        angle_dir = os.path.basename(os.path.dirname(image_path))
        angle_output_folder = os.path.join(output_folder, angle_dir)
        os.makedirs(angle_output_folder, exist_ok=True)
        output_path = os.path.join(angle_output_folder, os.path.basename(image_path))
        blended_image.save(output_path)
        
        # Extract both the sub_id and the group_id.
        sub_id, group_id = extract_image_ids(image_path)
        # Build a composite key to uniquely identify each image instance.
        composite_key = f"{sub_id}_{group_id}"
        
        if composite_key not in csv_data:
            csv_data[composite_key] = {
                "image_id": group_id,
                "sub_id": sub_id,
                "path_0": None,
                "path_90": None,
                "path_180": None,
                "path_270": None,
                "pixel_distribution": {}
            }
            
        csv_data[composite_key][f"path_{angle_dir}"] = output_path
        
        # Update the pixel distribution.
        for key, count in pixel_distribution.items():
            csv_data[composite_key]["pixel_distribution"][key] = csv_data[composite_key]["pixel_distribution"].get(key, 0) + count

@app.command()
def main(
        excel_file_path: Path = EXTERNAL_DATA_DIR / "macro_classes_with_colors.xlsx",
        image_path: Path = PROCESSED_DATA_DIR / "Bari",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    angle_dirs = ['0', '90', '180', '270']  # immagini in cartella flat, senza sottocartelle

    if image_path.exists():
        output_folder = image_path / 'masked_output'
        csv_path = image_path / 'output.csv'

    if excel_file_path.exists():
        df = pd.read_excel(excel_file_path)
        color_mapping = {
            label: ast.literal_eval(row['RGB'])
            for _, row in df.iterrows()
            for label in str(row['Original Labels']).split(', ')
        }
        macro_mapping = {
            label: row['New Class']
            for _, row in df.iterrows()
            for label in str(row['Original Labels']).split(', ')
        }

    processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-coco-panoptic")
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        "facebook/mask2former-swin-small-coco-panoptic"
    ).to(device)

    image_paths = get_image_paths(image_path, angle_dirs=angle_dirs)
    dataset = StreetViewDataset(image_paths, color_mapping, macro_mapping, processor)
    dataloader = DataLoader(
        dataset,
        #set to 1 (from 32) for testing on nvidia 2050 with 4gb vram
        batch_size=1,
        shuffle=False,
        #0 -> disabling parallel works.
        num_workers=0,
        collate_fn=custom_collate
    )

    csv_data = {}

    for batch in tqdm(dataloader, desc="Processing Images", unit="batch"):
        image_paths_batch, blended_images, pixel_distributions = process_batch(
            model, batch, color_mapping, macro_mapping, device, processor, output_folder
        )
        update_csv_data(image_paths_batch, blended_images, pixel_distributions, output_folder, csv_data)

    for data in csv_data.values():
        data["pixel_distribution"] = str(data["pixel_distribution"])

    df_csv = pd.DataFrame(list(csv_data.values()))
    df_csv.to_csv(csv_path, index=False)


    ''' non sono interessato alle coordinate
    merge_output_with_coordinates(csv_path, EXTERNAL_DATA_DIR / "merged_coordinates.csv",
                                  image_path / "output_with_coordinates.csv")
    '''


def print_all_classes(model):
    id2label = model.config.id2label
    print(f"Totale classi disponibili: {len(id2label)}")
    for class_id in sorted(id2label):
        print(f"{class_id:3d}: {id2label[class_id]}")


if __name__ == "__main__":
    #app()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        "facebook/mask2former-swin-small-coco-panoptic"
    ).to(device)

    print_all_classes(model)


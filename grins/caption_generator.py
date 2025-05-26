import os
import json
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration

# This dictionary provides cleaner and more natural language alternatives
# to class names extracted from the Mask2Former segmentation model.
# Some class labels (e.g., "tree-merged") include suffixes like "-merged" or "-other"
# that may confuse the language model or lead to hallucinations.
# The mapping below standardizes these labels to more human-friendly terms
# to improve the clarity and accuracy of caption generation.
label_natural = {
    "tree-merged": "tree",
    "fence-merged": "fence",
    "ceiling-merged": "ceiling",
    "sky-other-merged": "sky",
    "cabinet-merged": "cabinet",
    "table-merged": "table",
    "floor-other-merged": "floor",
    "pavement-merged": "pavement",
    "mountain-merged": "mountain",
    "grass-merged": "grass",
    "dirt-merged": "dirt",
    "paper-merged": "paper",
    "food-other-merged": "food",
    "building-other-merged": "building",
    "rock-merged": "rock",
    "wall-other-merged": "wall",
    "rug-merged": "rug",
    "wall-brick": "wall",
    "wall-stone": "wall",
    "wall-tile": "wall",
    "wall-wood": "wall",
    "window-blind": "window",
    "window-other": "window",
    "floor-wood": "floor",
    "door-stuff": "door",
    "mirror-stuff": "mirror",
    "water-other": "water",
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Model and processor (non quantized - linux required)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)
model.eval()

def generate_caption(crop: Image.Image, class_name: str):
    """
    Generates a natural language caption for a cropped region of the image,
    guided by a class-specific prompt.

    The prompt follows the format: "Describe the {class_name} in the image",
    helping the model to focus on the expected type of object or region.

    Parameters:
    ----------
    crop : PIL.Image.Image
        The image region cropped using the segmentation mask.

    class_name : str
        The semantic class name associated with the region (e.g., 'tree').

    Returns:
    -------
    str
        A caption describing the region, based on both image and class context.
    """
    #giving the class name as a prompt
    inputs = processor(images=crop, text=class_name,  return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=20)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

def run_ureca_on_segmented_data(root_dir: str, city_name: str, test:bool):
    """
    Processes each *_labels.json file, generates captions for each valid region,
    and saves a new JSON file enriched with captions.
    """
    labels_dir = Path(root_dir) / city_name / "masked_output" / "labels"
    images_dir = Path(root_dir) / city_name
    output_dir = Path(root_dir) / city_name / "ureca_output"
    output_dir.mkdir(exist_ok=True)

    all_label_files = sorted(labels_dir.glob("*.json"))
    if test:
        label_files = all_label_files[:10]
    else:
        label_files = all_label_files

    for label_file in tqdm(sorted(labels_dir.glob("*.json")), desc=f"Processing {city_name}"):
        #if already processed, skip.
        out_path = output_dir / label_file.name.replace(".json", "_ureca.json")
        if out_path.exists():
            print(f"[SKIP] already processed: {out_path.name}")
            continue

        try:
            with open(label_file, "r") as f:
                data = json.load(f)

            img_name = data["image"]
            angle = Path(img_name).stem.split("_")[-1]
            img_path = images_dir / angle / img_name

            if not img_path.exists():
                print(f"[WARNING] Immagine non trovata: {img_path}")
                continue

            image = Image.open(img_path).convert("RGB")

            valid_region_found = False

            for region in data["regions"]:
                x1, y1, x2, y2 = region["bbox"]
                width = x2 - x1
                height = y2 - y1

                if width >= 16 and height >= 16:
                    crop = image.crop((x1, y1, x2, y2))
                    try:
                        class_name = label_natural.get(region["class_name"], region["class_name"])
                        region["caption"] = generate_caption(crop, class_name)
                        valid_region_found = True
                    except Exception as e:
                        print(f"[ERROR] Caption failed for {label_file.name}, bbox={region['bbox']}: {e}")
                        region["caption"] = "Caption generation error"
                else:
                    region["caption"] = "Too small region"

            if valid_region_found:
                out_path = output_dir / label_file.name.replace(".json", "_ureca.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            else:
                print(f"[INFO] Nessuna regione valida in {label_file.name}, file non salvato.")

        except Exception as e:
            print(f"[ERROR] Errore su {label_file.name}: {e}")

if __name__ == "__main__":
    os.chdir(Path(__file__).parent.parent)
    cities = ["Bari", "Manhattan", "Shibuya"]
    TEST_MODE = True
    for city in cities:
        run_ureca_on_segmented_data("data/processed", city_name=city, test=TEST_MODE)


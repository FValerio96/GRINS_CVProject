from pathlib import Path
from PIL import Image
from tqdm import tqdm
import typer
import os


app = typer.Typer()


@app.command()
def resize_images(
        src_dir: Path = Path("data/raw/subset"),
        dst_dir: Path = Path("data/processed/cityscapes_subset"),
        width: int = 512,
        height: int = 256
):
    """
    Ridimensiona tutte le immagini in src_dir e le salva in dst_dir.

    Esegui con:
    python grins/prepare_images.py resize-images --src-dir <src> --dst-dir <dst> --width <w> --height <h>
    """
    dst_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted([f for f in src_dir.glob("*.png")])

    for img_path in tqdm(image_files, desc="Resizing images"):
        img = Image.open(img_path).convert("RGB")
        resized = img.resize((width, height))
        resized.save(dst_dir / img_path.name)

    typer.echo(f"✔ Salvate {len(image_files)} immagini in {dst_dir}")


def crop_bottom(img: Image.Image, bottom: int = 40) -> Image.Image:
    """
    Crop the bottom band to remove watermark.

    Args:
        img (PIL.Image): Original image.
        bottom (int): Height in pixels to crop from the bottom.

    Returns:
        PIL.Image: cropped image.
    """
    os.chdir(Path(__file__).parent.parent)
    width, height = img.size
    return img.crop((0, 0, width, height - bottom))

def process_city_images(
        city_name: str,
        raw_root: Path = Path("data/raw"),
        processed_root: Path = Path("data/processed"),
        angles: list[str] = ['0', '90', '180', '270'],
        crop_bottom_px: int = 40
):
    """
    Prepare images of a city by cropping the watermarks at the bottom
    and copying them into the processed/{city}/{angle}/ structure.
    """
    src_city = raw_root / city_name
    dst_city = processed_root / city_name

    for angle in angles:
        src_dir = src_city / angle
        dst_dir = dst_city / angle
        dst_dir.mkdir(parents=True, exist_ok=True)

        image_files = sorted([f for f in src_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])

        for img_path in tqdm(image_files, desc=f"Processando {city_name} - {angle}°"):
            img = Image.open(img_path).convert("RGB")
            cropped = crop_bottom(img, bottom=crop_bottom_px)
            cropped.save(dst_dir / img_path.name)

    typer.echo(f"✔ Completato: {city_name}")


if __name__ == "__main__":
    cities_name = ["Manhattan", "Bari"]
    os.chdir(Path(__file__).parent.parent)
    for city in cities_name:
        process_city_images(city_name=city)

    #app()

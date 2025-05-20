from pathlib import Path
from PIL import Image
from tqdm import tqdm
import typer

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


if __name__ == "__main__":
    app()

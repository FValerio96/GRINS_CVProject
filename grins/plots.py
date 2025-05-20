from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from loguru import logger
import typer

from grins.config import PROCESSED_DATA_DIR


class Plotter:
    def __init__(self, input_csv: Path):
        self.input_csv = input_csv
        logger.info(f"Caricamento CSV da: {input_csv}")
        self.df = pd.read_csv(input_csv)
        self._parse_pixel_distribution()
        logger.success("CSV caricato e distribuzione pixel parsata.")

    def _parse_pixel_distribution(self):
        """
        Parsing sicuro della colonna `pixel_distribution` in veri dizionari.
        """
        def parse_dist(dist_str):
            try:
                pairs = re.findall(r"'([^']+)': np\.int64\((\d+)\)", dist_str)
                return {key: int(val) for key, val in pairs}
            except Exception as e:
                logger.error(f"Errore parsing: {e}")
                return {}

        self.df['parsed_distribution'] = self.df['pixel_distribution'].apply(parse_dist)

    def plot_pixel_distribution(self, output_path: Path):
        """
        Genera un barplot aggregato del numero di pixel per categoria.
        """
        logger.info("Generazione grafico distribuzione pixel per categoria...")

        category_pixel_counts = {}
        for dist in self.df['parsed_distribution']:
            for category, pixel_count in dist.items():
                category_pixel_counts[category] = category_pixel_counts.get(category, 0) + pixel_count

        category_df = pd.DataFrame(list(category_pixel_counts.items()), columns=["category", "pixel_count"])
        category_df = category_df.sort_values("pixel_count", ascending=False)

        plt.figure(figsize=(12, 6))
        sns.barplot(data=category_df, x="pixel_count", y="category")
        plt.title("Pixel Count per Category (Aggregated)")
        plt.xlabel("Total Pixels")
        plt.ylabel("Category")
        plt.tight_layout()
        plt.savefig(output_path)
        logger.success(f"Grafico salvato in: {output_path}")

# Typer CLI
app = typer.Typer()

@app.command()
def main(
    input_csv: Path = PROCESSED_DATA_DIR / "cityscapes_subset" / "output.csv",
    output_plot:  Path = PROCESSED_DATA_DIR / "cityscapes_subset" / "pixel_distibution.png",
):
    """
    Genera un grafico aggregato dei pixel per categoria a partire da un file CSV.
    """
    plotter = Plotter(input_csv)
    plotter.plot_pixel_distribution(output_plot)

if __name__ == "__main__":
    app()

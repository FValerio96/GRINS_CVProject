import heapq
import os
import torch
import clip
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import List, Tuple
from pathlib import Path
import json

WHITE_EMBEDDINGS_PATH = "grins/modeling/clip_white_embedding_full.npy"
IGNORED_CLASSES = {
    "sky-other-merged", "grass-merged", "dirt-merged", "water-other",
    "tree-merged", "rock-merged", "mountain-merged",  # natura generica
    "rug-merged", "mirror-stuff", "ceiling-merged", "floor-other-merged", "floor-wood",
    "paper-merged", "food-other-merged", "cabinet-merged"  # interni non strutturali
}

class CLIPRetriever:


    def __init__(self, base_dir: str, city: str, embedding_save_path: str = "clip_embeddings.npy",
                 device: str = "cuda"):
        self.city = city
        self.base_dir = base_dir  # es: "data/processed"
        self.image_root = os.path.join(base_dir, city)
        self.labels_dir = os.path.join(self.image_root, "masked_output", "labels")
        self.masks_root = os.path.join(self.image_root, "masked_output", "masks")
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device, download_root="G:/Flavio/clip_cache")
        self.embedding_save_path = embedding_save_path
        self.embeddings = {}

    def _load_json_and_mask(self, label_file: str):
        json_path = os.path.join(self.labels_dir, label_file)
        with open(json_path, "r") as f:
            data = json.load(f)

        image_name = data["image"]  # es. image_0_1_heading_90.jpg
        mask_file = image_name.replace(".jpg", ".npy")

        # Estrai l'angolazione dalla fine del nome
        heading = image_name.split("_")[-1].replace(".jpg", "")  # es. "90"

        # Costruisci il path completo alla maschera
        mask_path = os.path.join(self.base_dir, self.city, "masked_output/masks", mask_file)

        # Controllo esistenza per debug
        if not os.path.isfile(mask_path):
            raise FileNotFoundError(f"❌ Maschera non trovata: {mask_path}")

        mask = np.load(mask_path)
        return data, mask

    def _get_region_crop(self, image: Image.Image, mask: np.ndarray, bbox: list, class_id: int) -> Image.Image:
        """
        Estrae un crop RGB dell'immagine in corrispondenza della bounding box, isolando
        solo i pixel della classe indicata con sfondo bianco.

        Args:
            image (PIL.Image): immagine RGB originale.
            mask (np.ndarray): maschera semantica segmentata (HxW).
            bbox (list): bounding box [x1, y1, x2, y2].
            class_id (int): id della classe da isolare nella maschera.

        Returns:
            PIL.Image: crop RGB con oggetto isolato su sfondo bianco.
        """
        x1, y1, x2, y2 = bbox
        region_mask = (mask[y1:y2, x1:x2] == class_id).astype(np.uint8) * 255
        crop = image.crop((x1, y1, x2, y2)).convert("RGBA")
        white_bg = Image.new("RGBA", crop.size, (255, 255, 255, 255))
        mask_img = Image.fromarray(region_mask).resize(crop.size).convert("L")
        composite = Image.composite(crop, white_bg, mask_img)
        return composite.convert("RGB")

    def _is_visually_informative(self, image: Image.Image, min_pixel_sum: int = 1000, min_height: int = 20) -> bool:
        np_img = np.array(image)

        # Filtro 1: immagine quasi bianca (pochi pixel attivi)
        if np_img.sum() < min_pixel_sum:
            return False

        # Filtro 2: crop troppo schiacciato
        _, h = image.size
        if h < min_height:
            return False

        return True

    def compute_embeddings(self, limit: int = None):
        """
        Calcola gli embeddings delle regioni da immagini segmentate.

        Args:
            limit (int, optional): massimo numero di immagini (JSON) da processare.
                                   Se None, processa tutte le immagini.
        """
        if os.path.exists(self.embedding_save_path):
            self.embeddings = np.load(self.embedding_save_path, allow_pickle=True).item()
            print(f"✅ Caricati {len(self.embeddings)} embeddings esistenti da {self.embedding_save_path}")

        json_files = [f for f in os.listdir(self.labels_dir) if f.endswith(".json")]
        if limit is not None:
            json_files = json_files[:limit]

        for label_file in tqdm(json_files, desc=f"📂 Processing {self.city}", unit="file"):
            data, mask = self._load_json_and_mask(label_file)

            image_name = Path(data["image"]).name
            base_name = Path(image_name).stem
            heading = image_name.split("_")[-1].replace(".jpg", "")
            image_path = os.path.join(self.image_root, heading, image_name)

            if not os.path.exists(image_path):
                print(f"⚠️  Immagine non trovata: {image_path}")
                continue

            image = Image.open(image_path).convert("RGB")

            for idx, region in enumerate(data["regions"]):
                class_name = region.get("class_name", "").strip().lower()
                if class_name in (cls.lower() for cls in IGNORED_CLASSES):
                    print(f"☁️  Regione cielo ignorata: {class_name}")
                    continue

                unique_name = f"{self.city}_{base_name}_region_{idx}"
                if unique_name in self.embeddings:
                    continue

                bbox = region["bbox"]
                x1, y1, x2, y2 = bbox
                region_area = (x2 - x1) * (y2 - y1)

                if x2 - x1 <= 1 or y2 - y1 <= 1:
                    print(f"⚠️  Bbox non valida per {image_name}, regione {idx}, bbox={bbox}")
                    continue

                try:
                    class_id = region["class_id"]
                    region_crop = self._get_region_crop(image, mask, bbox, class_id)

                    if not self._is_visually_informative(region_crop):
                        print(f"⚠️  Regione non informativa: {unique_name}")
                        continue

                    print(f"🖼️  {unique_name} → crop size: {region_crop.size}, area: {region_area}")

                    input_tensor = self.preprocess(region_crop).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        embedding = self.model.encode_image(input_tensor)
                        embedding /= embedding.norm(dim=-1, keepdim=True)
                        self.embeddings[unique_name] = embedding.cpu().numpy().flatten()

                except Exception as e:
                    print(f"❌ Errore su {unique_name}: {e}")
                    continue

        np.save(self.embedding_save_path, self.embeddings)
        print(f"✅ Salvati {len(self.embeddings)} embeddings in {self.embedding_save_path}")

    def remove_useless_embeddings(self, white_embed_path: str, threshold: float = 0.9999):
        """
        Rimuove gli embeddings troppo simili a quello dell'immagine bianca.
        """
        if not os.path.exists(white_embed_path):
            raise FileNotFoundError(f"❌ File embedding bianco non trovato: {white_embed_path}")

        white_embed = np.load(white_embed_path)
        to_remove = []

        for key, emb in self.embeddings.items():
            sim = np.dot(emb, white_embed) / (np.linalg.norm(emb) * np.linalg.norm(white_embed))
            if sim > threshold:
                print(f"  {key} rimosso: sim={sim:.5f}")
                #if too similare will be removed, i save key which is used for identify embeddings
                to_remove.append(key)

        for key in to_remove:
            del self.embeddings[key]

        np.save(self.embedding_save_path, self.embeddings)
        print(f"✅ Rimozione completata. Rimasti {len(self.embeddings)} embeddings.")

    def load_embeddings(self):
        self.embeddings = np.load(self.embedding_save_path, allow_pickle=True).item()

    def retrieve_between_cities(self, city1: str, city2: str, embedding_paths: dict, top_k: int = 5,
                                resume: bool = True, save_path: str = "retrieval_results") -> List[
        Tuple[str, str, float]]:

        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, f"{city1}_vs_{city2}_top1_per_region.json")
        final_topk_file = os.path.join(save_path, f"{city1}_vs_{city2}_top{top_k}_global.json")
        done_file = os.path.join(save_path, f"{city1}_vs_{city2}_done_keys.txt")

        emb1 = np.load(embedding_paths[city1], allow_pickle=True).item()
        emb2 = np.load(embedding_paths[city2], allow_pickle=True).item()

        all_top1_matches = []
        done_names1 = set()

        # Riprendi se esistono
        if resume and os.path.exists(save_file):
            with open(save_file, "r") as f:
                all_top1_matches = json.load(f)
            print(f"🔁 Ripresi {len(all_top1_matches)} top-1 già salvati da {save_file}")

        if resume and os.path.exists(done_file):
            with open(done_file, "r") as f:
                done_names1 = set(line.strip() for line in f)
            print(f"🔁 Ripresi {len(done_names1)} elementi già processati da {done_file}")

        for current_name1, vec1 in tqdm(emb1.items(), desc=f"Comparing {city1} vs {city2}"):
            if current_name1 in done_names1:
                continue

            best_sim = -1.0
            best_match = None

            for current_name2, vec2 in emb2.items():
                sim = float(np.dot(vec1, vec2))
                if sim > best_sim:
                    best_sim = sim
                    best_match = current_name2

            all_top1_matches.append({
                "sim": best_sim,
                "name1": current_name1,
                "name2": best_match
            })

            # Salva progressivamente
            with open(save_file, "w") as f:
                json.dump(all_top1_matches, f, indent=2)

            with open(done_file, "a") as f:
                f.write(f"{current_name1}\n")

        # Ora estrai top-k globali
        topk_global = sorted(all_top1_matches, key=lambda x: x["sim"], reverse=True)[:top_k]
        with open(final_topk_file, "w") as f:
            json.dump(topk_global, f, indent=2)
        print(f"✅ Top-{top_k} globali salvati in {final_topk_file}")

        return [(entry["name1"], entry["name2"], entry["sim"]) for entry in topk_global]

    def extract_class_ids(self, city: str) -> dict:
        retriever = CLIPRetriever(base_dir, city, "dummy.npy")  # dummy path
        labels_dir = os.path.join(base_dir, city, "masked_output", "labels")
        name_to_class = {}

        for label_file in os.listdir(labels_dir):
            if not label_file.endswith(".json"):
                continue
            json_path = os.path.join(labels_dir, label_file)
            with open(json_path, "r") as f:
                data = json.load(f)
            image_name = Path(data["image"]).stem
            for idx, region in enumerate(data["regions"]):
                key = f"{city}_{image_name}_region_{idx}"
                name_to_class[key] = region.get("class_id")
        return name_to_class

    def retrieve_between_cities_with_region_class_matching(self, city1: str, city2: str, embedding_paths: dict, top_k: int = 5,
                                resume: bool = True, save_path: str = "retrieval_results") -> List[
        Tuple[str, str, float]]:
        os.makedirs(save_path, exist_ok=True)
        # mapping embedding → class_id
        class_map1 = self.extract_class_ids(city1)
        class_map2 = self.extract_class_ids(city2)

        save_file = os.path.join(save_path, f"{city1}_vs_{city2}_top1_per_region_with_class_matching.json")
        final_topk_file = os.path.join(save_path, f"{city1}_vs_{city2}_top{top_k}_global_with_class_matching.json")
        done_file = os.path.join(save_path, f"{city1}_vs_{city2}_done_keys_with_class_matching.txt")

        emb1 = np.load(embedding_paths[city1], allow_pickle=True).item()
        emb2 = np.load(embedding_paths[city2], allow_pickle=True).item()

        all_top1_matches = []
        done_names1 = set()

        # Riprendi se esistono
        if resume and os.path.exists(save_file):
            with open(save_file, "r") as f:
                all_top1_matches = json.load(f)
            print(f"🔁 Ripresi {len(all_top1_matches)} top-1 già salvati da {save_file}")

        if resume and os.path.exists(done_file):
            with open(done_file, "r") as f:
                done_names1 = set(line.strip() for line in f)
            print(f"🔁 Ripresi {len(done_names1)} elementi già processati da {done_file}")

        for current_name1, vec1 in tqdm(emb1.items(), desc=f"Comparing {city1} vs {city2}"):
            if current_name1 in done_names1:
                continue

            best_sim = -1.0
            best_match = None

            class_id1 = class_map1.get(current_name1)
            if class_id1 is None or class_id1 == "car":
                continue  # se non trovi la classe, salta

            for current_name2, vec2 in emb2.items():
                class_id2 = class_map2.get(current_name2)
                if class_id2 is None or class_id2 != class_id1:
                    continue  # classi diverse → salta

                sim = float(np.dot(vec1, vec2))
                if sim > best_sim:
                    best_sim = sim
                    best_match = current_name2

            all_top1_matches.append({
                "sim": best_sim,
                "name1": current_name1,
                "name2": best_match
            })

            # Salva progressivamente
            with open(save_file, "w") as f:
                json.dump(all_top1_matches, f, indent=2)

            with open(done_file, "a") as f:
                f.write(f"{current_name1}\n")

        # Ora estrai top-k globali
        topk_global = sorted(all_top1_matches, key=lambda x: x["sim"], reverse=True)[:top_k]
        with open(final_topk_file, "w") as f:
            json.dump(topk_global, f, indent=2)
        print(f"✅ Top-{top_k} globali salvati in {final_topk_file}")

        return [(entry["name1"], entry["name2"], entry["sim"]) for entry in topk_global]



    def build_topk_from_done(city1: str, city2: str, embedding_paths: dict, done_file: str,
                               top_k: int = 5, output_file: str = "top5_clean.json"):
        emb1 = np.load(embedding_paths[city1], allow_pickle=True).item()
        emb2 = np.load(embedding_paths[city2], allow_pickle=True).item()

        with open(done_file, "r") as f:
            done_names1 = [line.strip() for line in f]

        heap = []

        for name1 in tqdm(done_names1, desc=f"🔁 Rebuild {city1} vs {city2}"):
            vec1 = emb1[name1]
            for name2, vec2 in emb2.items():
                sim = float(np.dot(vec1, vec2))
                item = (sim, name1, name2)
                if len(heap) < top_k:
                    heapq.heappush(heap, item)
                else:
                    heapq.heappushpop(heap, item)

        top_results = sorted(heap, reverse=True)

        with open(output_file, "w") as f:
            json.dump([
                {"sim": sim, "name1": n1, "name2": n2}
                for sim, n1, n2 in top_results
            ], f, indent=2)

        print(f"✅ Ricostruito top-{top_k} salvato in {output_file}")

def print_city_embeddings(embedding_path: str, max_per_city: int = None):
    if not os.path.exists(embedding_path):
        print(f"❌ File non trovato: {embedding_path}")
        return

    embeddings = np.load(embedding_path, allow_pickle=True).item()
    print(f"✅ Caricati {len(embeddings)} embeddings da {embedding_path}")

    for i, (key, vector) in enumerate(embeddings.items()):
        print(f"\n🔹 {key} → {vector[:50]}...")  # Mostra solo i primi 10 valori del vettore
        if max_per_city and i >= max_per_city - 1:
            break

if __name__ == "__main__":
    import shutil

    os.chdir(Path(__file__).resolve().parent.parent.parent)
    print("📂 Working directory:", os.getcwd())
    base_dir = "data/processed"
    save_path = "retrieval_results"
    top_k = 5
    debug_crop_dir = "debug_crops"
    cities = ["Bari", "Shibuya", "Manhattan"]
    city = "Bari"
    os.makedirs(debug_crop_dir, exist_ok=True)
    embedding_paths = {
        city: os.path.join(base_dir, city, f"{city.lower()}_clip.npy") for city in cities
    }

    #Bari computing embeddings

    retriever = CLIPRetriever(
        base_dir=base_dir,
        city=city,
        embedding_save_path=embedding_paths[city],
    )
    '''
    retriever.compute_embeddings()
    print("avvio pulizia embeddings")
    retriever.remove_useless_embeddings(WHITE_EMBEDDINGS_PATH)

    #shibuya computing embeddings.
    city = "Shibuya"
    retriever = CLIPRetriever(
        base_dir=base_dir,
        city=city,
        embedding_save_path=embedding_paths[city],
    )
    retriever.compute_embeddings()
    print("avvio pulizia embeddings")
    retriever.remove_useless_embeddings(WHITE_EMBEDDINGS_PATH)

    results = retriever.retrieve_between_cities(
        city1="Bari",
        city2="Shibuya",
        embedding_paths=embedding_paths,
        top_k=5,
        resume=False
    )
    '''
    retriever.retrieve_between_cities_with_region_class_matching("Bari", "Shibuya", embedding_paths=embedding_paths, top_k=5, resume=False)









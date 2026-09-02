from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
IMAGE_DIR = DATA_DIR / "images"
CSV_PATH = DATA_DIR / "candidate_dataset.csv"


class JewelleryMatcher:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading CLIP model on: {self.device}")

        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

        self.model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        ).to(self.device)

        self.model.eval()

        # Load inventory
        self.dataset = pd.read_csv(CSV_PATH)

        # Only earrings are candidates for recommendation
        self.earrings = self.dataset[
            self.dataset["product_type"].str.lower() == "earrings"
        ].copy()

        if len(self.earrings) == 0:
            raise ValueError("No earrings found in candidate_dataset.csv")

        print(f"Loaded {len(self.earrings)} earrings")

        # Precompute earring embeddings once
        self.earring_embeddings = self._build_earring_embeddings()

        print("Earring embeddings created successfully.")

    def _load_image(self, image_path: Path) -> Image.Image:
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        return Image.open(image_path).convert("RGB")

    def _get_embedding(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            vision_outputs = self.model.vision_model(**inputs)

            # Get the pooled image representation
            pooled_output = vision_outputs.pooler_output

            # Project it into CLIP's embedding space
            image_features = self.model.visual_projection(pooled_output)

        # Normalize embedding
        image_features = image_features / image_features.norm(
            dim=-1,
            keepdim=True
        )

        return image_features.cpu().numpy()[0]

    def _build_earring_embeddings(self) -> np.ndarray:
        embeddings = []

        for _, row in self.earrings.iterrows():
            image_path = IMAGE_DIR / row["image_file"]

            print(f"Embedding: {row['image_file']}")

            image = self._load_image(image_path)
            embedding = self._get_embedding(image)

            embeddings.append(embedding)

        return np.array(embeddings)

    def recommend(
        self,
        necklace_image: Image.Image,
        top_k: int = 3
    ) -> List[Dict]:

        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        top_k = min(top_k, len(self.earrings))

        # Generate embedding for uploaded necklace
        necklace_embedding = self._get_embedding(necklace_image)

        # Since both embeddings are normalized,
        # dot product = cosine similarity
        similarities = np.dot(
            self.earring_embeddings,
            necklace_embedding
        )

        # Highest similarity first
        ranked_indices = np.argsort(similarities)[::-1][:top_k]

        recommendations = []

        for index in ranked_indices:
            row = self.earrings.iloc[index]

            recommendations.append({
                "id": row["id"],
                "product_type": row["product_type"],
                "image_file": row["image_file"],
                "image_url": f"/images/{row['image_file']}",
                "similarity": round(
                    float(similarities[index]),
                    4
                )
            })

        return recommendations
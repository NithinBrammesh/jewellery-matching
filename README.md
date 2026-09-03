# Jewellery Matching API

A computer-vision prototype that recommends matching earrings for a selected necklace from a fixed jewellery inventory.

The system treats jewellery matching as an **image retrieval problem**. A pretrained CLIP vision model converts the input necklace and the available earrings into visual embeddings. Cosine similarity is then used to rank the earrings by visual similarity, and the top-K recommendations are returned.

## 🎥 Demo

A complete project walkthrough and demonstration is available here:

**[▶️ Watch the Project Demo](https://www.tella.tv/video/jewelry-matching-assessment-project-h0pr)**

The demo covers the project approach, API functionality, jewellery matching flow, and recommendation results.

---

## Features

- Accepts a necklace image as input through a REST API.
- Uses the provided `candidate_dataset.csv` as the product inventory.
- Restricts recommendations to the 15 provided earrings.
- Uses a pretrained CLIP model; no model training is required.
- Precomputes earring embeddings when the application starts.
- Uses cosine similarity to rank visual matches.
- Returns product ID, image filename, image URL, and similarity score.
- Provides interactive Swagger API documentation.
- Provides a health-check endpoint.

---

## Dataset

The provided inventory contains:

### Necklaces

- `N01` → `Nck_1.jpg`
- `N02` → `Nck_2.jpg`
- `N03` → `Nck_3.jpg`
- `N04` → `Nck_4.jpg`
- `N05` → `Nck_5.jpg`

### Earrings

- `E01` → `Ear_1.jpg`
- `E02` → `Ear_2.jpg`
- `E03` → `Ear_3.jpg`
- `...`
- `E015` → `Ear_15.jpg`

The inventory metadata is stored in:

```text
data/candidate_dataset.csv
```

Images are stored in:

```text
data/images/
```

---

## Approach

### 1. Load the inventory

The application reads `candidate_dataset.csv` and filters the inventory to products where:

```text
product_type = Earrings
```

This guarantees that recommendations come only from the provided earring inventory.

### 2. Generate earring embeddings

At application startup, each of the 15 earring images is passed through the pretrained:

```text
openai/clip-vit-base-patch32
```

model.

The resulting image embeddings are normalized and kept in memory.

This avoids recomputing all 15 earring embeddings for every API request.

### 3. Generate the necklace embedding

When a user uploads a necklace image, the image is converted into an embedding using the same CLIP model.

### 4. Calculate visual similarity

The normalized necklace embedding is compared with the precomputed earring embeddings.

Cosine similarity is used as the similarity measure.

Because the embeddings are normalized, cosine similarity can be calculated efficiently using the dot product.

### 5. Rank and return recommendations

The earrings are sorted from highest similarity to lowest similarity.

The API returns the requested number of top matches using the `top_k` parameter.

The default value of `top_k` is `3`.

---

## Architecture

```text
                Necklace Image
                       |
                       v
                FastAPI Endpoint
                       |
                       v
                  CLIP Encoder
                       |
                       v
               Necklace Embedding
                       |
                       v
        Compare with 15 Earring
                Embeddings
                       |
                       v
             Cosine Similarity
                       |
                       v
             Sort Highest First
                       |
                       v
             Top-K Recommendations
```

---

## Technology Stack

- **Python** - Application development
- **FastAPI** - REST API
- **PyTorch** - Model inference
- **Hugging Face Transformers** - CLIP model and processor
- **CLIP (`openai/clip-vit-base-patch32`)** - Pretrained visual embeddings
- **Pillow** - Image loading and processing
- **NumPy** - Embedding and similarity calculations
- **Pandas** - Inventory and CSV handling
- **Uvicorn** - ASGI server

---

## Project Structure

```text
jewellery-matching/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── matcher.py
│   └── utils.py
│
├── data/
│   ├── images/
│   │   ├── Ear_1.jpg
│   │   ├── ...
│   │   ├── Ear_15.jpg
│   │   ├── Nck_1.jpg
│   │   ├── ...
│   │   └── Nck_5.jpg
│   │
│   └── candidate_dataset.csv
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd jewellery-matching
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
```

### 3. Activate the virtual environment

#### Linux/macOS

```bash
source venv/bin/activate
```

#### Windows

```powershell
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run the API

From the project root:

```bash
uvicorn app.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

### Swagger Documentation

Interactive Swagger documentation is available at:

```text
http://127.0.0.1:8000/docs
```

You can use Swagger to upload a necklace image and test the recommendation API directly.

---

## First Startup

On the first startup, the pretrained CLIP model is downloaded and loaded.

The application then generates embeddings for all 15 earrings.

Subsequent requests reuse these precomputed earring embeddings while only the uploaded necklace needs to be embedded.

This reduces unnecessary computation for every recommendation request.

---

## API

### `POST /recommend`

Recommends matching earrings for an uploaded necklace image.

### Parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| `file` | Image file | Yes | Necklace image |
| `top_k` | Integer | No | Number of recommendations, from 1 to 15. Default: 3 |

### Example Using Swagger

1. Open `/docs`.
2. Expand `POST /recommend`.
3. Click **Try it out**.
4. Upload a necklace image such as `Nck_1.jpg`.
5. Set `top_k` to `3`.
6. Click **Execute**.
7. View the recommended earrings and similarity scores in the response.

---

## Example Response

```json
{
  "input_image": "Nck_1.jpg",
  "recommendations": [
    {
      "id": "E08",
      "product_type": "Earrings",
      "image_file": "Ear_8.jpg",
      "image_url": "/images/Ear_8.jpg",
      "similarity": 0.8696
    },
    {
      "id": "E01",
      "product_type": "Earrings",
      "image_file": "Ear_1.jpg",
      "image_url": "/images/Ear_1.jpg",
      "similarity": 0.8664
    },
    {
      "id": "E012",
      "product_type": "Earrings",
      "image_file": "Ear_12.jpg",
      "image_url": "/images/Ear_12.jpg",
      "similarity": 0.8496
    }
  ]
}
```

The exact recommendations and similarity scores can vary depending on the model/library version and input image.

---

## Image URLs

The API serves the provided jewellery images through:

```text
/images/{image_file}
```

For example:

```text
http://127.0.0.1:8000/images/Ear_8.jpg
```

---

## Health Check

The API also provides a health-check endpoint:

```text
GET /health
```

Example response:

```json
{
  "status": "healthy",
  "earrings_available": 15
}
```

This can be used to verify that the API is running and that the earring inventory has been loaded successfully.

---

## Design Decisions

### Why CLIP?

The assignment requires visual matching and allows the use of pretrained image embeddings.

CLIP provides a practical way to represent images as vectors without training a new model from scratch.

It allows the system to compare the visual characteristics of the input necklace against the available earrings.

### Why cosine similarity?

Cosine similarity measures how close two embedding vectors are in direction.

For normalized embeddings, cosine similarity can be calculated efficiently using their dot product.

This makes it suitable for comparing the necklace embedding against the precomputed earring embeddings.

### Why precompute earring embeddings?

The inventory contains a fixed set of 15 earrings.

Their embeddings do not change between requests, so computing them once during application startup avoids repeating the same model inference for every API request.

Only the uploaded necklace image needs to be processed for each request.

### Why filter by product type?

The task requires recommendations to be earrings selected from the provided inventory.

Filtering the CSV before matching ensures that necklaces are never returned as recommendations.

---

## Limitations

This is a prototype rather than a production recommendation system.

The visual similarity score does not necessarily represent jewellery styling compatibility perfectly.

CLIP captures general visual similarity, but matching jewellery can also depend on specific attributes such as:

- Metal colour
- Gemstone colour
- Shape
- Design pattern
- Size
- Style and formality
- Occasion

With a larger real-world inventory, the system could be improved by combining CLIP embeddings with domain-specific visual features, colour analysis, metadata, or a jewellery-specific model.

---

## Future Improvements

Possible improvements include:

- Add a lightweight frontend for image upload and visual recommendations.
- Display recommended earring images directly in the UI.
- Store embeddings in a vector database for a larger inventory.
- Combine semantic image embeddings with colour and shape features.
- Add product metadata such as metal type, gemstone, style, and price.
- Evaluate recommendation quality using human-labelled matching pairs.
- Add caching and batch inference for larger inventories.
- Add GPU-based inference for higher throughput.
- Add automated tests for API endpoints and recommendation logic.
- Containerize the application using Docker for easier deployment.

---

## Assignment Requirements Covered

- [x] User can provide a necklace image as input.
- [x] Visual similarity is used for matching.
- [x] Recommendations come from the provided earring inventory.
- [x] Recommendations are ranked using similarity scores.
- [x] Top matching earrings are returned.
- [x] Earring embeddings are precomputed for efficient inference.
- [x] REST API is implemented using FastAPI.
- [x] Interactive Swagger documentation is available.
- [x] Health-check endpoint is available.
- [x] Code is provided in a GitHub-ready project.
- [x] README documents the approach and technologies used.
- [x] Project demo video is provided.

---

## Demo

🎥 **Project Walkthrough:**

[Watch the Jewellery Matching Project Demo](https://www.tella.tv/video/jewelry-matching-assessment-project-h0pr)

The demo provides a walkthrough of the implementation and demonstrates the jewellery matching API.

---

## Author

**Nithin B**

Backend / AI Engineering
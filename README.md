# Jewellery Matching API

A simple computer-vision prototype that recommends matching earrings for a selected necklace from a fixed jewellery inventory.

The system treats jewellery matching as an **image retrieval problem**. A pretrained CLIP vision model converts the input necklace and the available earrings into visual embeddings. Cosine similarity is then used to rank the earrings by visual similarity, and the top-K recommendations are returned.

## Features

- Accepts a necklace image as input through a REST API.
- Uses the provided `candidate_dataset.csv` as the product inventory.
- Restricts recommendations to the 15 provided earrings.
- Uses a pretrained CLIP model; no model training is required.
- Precomputes earring embeddings when the application starts.
- Uses cosine similarity to rank visual matches.
- Returns product ID, image filename, image URL, and similarity score.
- Provides interactive Swagger API documentation.

## Dataset

The provided inventory contains:

- **5 necklaces**
  - `N01` → `Nck_1.jpg`
  - `N02` → `Nck_2.jpg`
  - `N03` → `Nck_3.jpg`
  - `N04` → `Nck_4.jpg`
  - `N05` → `Nck_5.jpg`
- **15 earrings**
  - `E01` → `Ear_1.jpg`
  - ...
  - `E015` → `Ear_15.jpg`

The inventory metadata is stored in:

```text
data/candidate_dataset.csv
```

Images are stored in:

```text
data/images/
```

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

When a user uploads a necklace image, the image is converted to an embedding using the same CLIP model.

### 4. Calculate visual similarity

The normalized necklace embedding is compared with the precomputed earring embeddings.

Cosine similarity is used as the similarity measure.

Because the embeddings are normalized, the implementation can calculate cosine similarity efficiently using the dot product.

### 5. Rank and return recommendations

The earrings are sorted from highest similarity to lowest similarity.

The API returns the requested number of top matches (`top_k`), with a default of 3.

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

## Technology Stack

- **Python**
- **FastAPI** - REST API
- **PyTorch** - model inference
- **Hugging Face Transformers** - CLIP model and processor
- **CLIP (`openai/clip-vit-base-patch32`)** - pretrained visual embeddings
- **Pillow** - image loading and processing
- **NumPy** - embedding and similarity calculations
- **Pandas** - inventory/CSV handling
- **Uvicorn** - ASGI server

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

Linux/macOS:

```bash
source venv/bin/activate
```

Windows:

```powershell
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## Run the API

From the project root:

```bash
uvicorn app.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

### First startup

On the first startup, the pretrained CLIP model is downloaded and loaded.

The application then generates embeddings for all 15 earrings.

Subsequent requests reuse these precomputed earring embeddings while only the uploaded necklace needs to be embedded.

## API

### `POST /recommend`

Recommends matching earrings for an uploaded necklace image.

#### Parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| `file` | Image file | Yes | Necklace image |
| `top_k` | Integer | No | Number of recommendations, from 1 to 15. Default: 3 |

#### Example

Using Swagger:

1. Open `/docs`.
2. Expand `POST /recommend`.
3. Click **Try it out**.
4. Upload a necklace image such as `Nck_1.jpg`.
5. Set `top_k` to `3`.
6. Click **Execute**.

### Example response

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

The exact recommendations and similarity scores can vary depending on the model/library version and input.

## Image URLs

The API serves the provided jewellery images through:

```text
/images/{image_file}
```

For example:

```text
http://127.0.0.1:8000/images/Ear_8.jpg
```

## Health Check

The API also provides:

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

## Design Decisions

### Why CLIP?

The assignment asks for visual matching and explicitly allows pretrained image embeddings. CLIP provides a practical way to represent images as vectors without training a new model from scratch.

### Why cosine similarity?

Cosine similarity measures how close two embedding vectors are in direction. For normalized embeddings, it can be calculated efficiently using their dot product.

### Why precompute earring embeddings?

The inventory contains a fixed set of 15 earrings. Their embeddings do not change between requests, so computing them once at startup reduces repeated inference work during API requests.

### Why filter by product type?

The task requires recommendations to be earrings selected from the provided inventory. Filtering the CSV before matching ensures that necklaces are never returned as recommendations.

## Limitations

This is a prototype rather than a production recommendation system.

The visual similarity score does not necessarily represent jewellery styling compatibility perfectly. CLIP captures general visual similarity, but matching jewellery can also depend on specific attributes such as:

- metal colour
- gemstone colour
- shape
- design pattern
- size
- style/formality

With a larger real-world inventory, the system could be improved by combining CLIP embeddings with domain-specific visual features, colour analysis, metadata, or a jewellery-specific model.

## Future Improvements

Possible improvements include:

- Add a lightweight frontend for image upload and visual recommendations.
- Display recommended earring images directly in the UI.
- Store embeddings in a vector database for a larger inventory.
- Combine semantic image embeddings with colour and shape features.
- Add product metadata such as metal type, gemstone, style, and price.
- Evaluate recommendation quality using human-labelled matching pairs.
- Add caching and batch inference for larger inventories.

## Assignment Requirements Covered

- [x] User can provide a necklace image as input.
- [x] Visual similarity is used for matching.
- [x] Recommendations come from the provided earring inventory.
- [x] Top matching earrings are returned.
- [x] Code is provided in a GitHub-ready project.
- [x] README documents the approach and technologies used.

## Author

**Nithin B**

Backend / AI Engineering
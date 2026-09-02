from io import BytesIO

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from PIL import Image

from app.matcher import JewelleryMatcher


app = FastAPI(
    title="Jewellery Matching API",
    description=(
        "Recommends matching earrings for a selected necklace "
        "using visual image similarity."
    ),
    version="1.0.0"
)


# Serve the provided jewellery images
app.mount(
    "/images",
    StaticFiles(directory="data/images"),
    name="images"
)


# Load model and precompute earring embeddings when API starts
matcher = JewelleryMatcher()


@app.get("/")
def root():
    return {
        "message": "Jewellery Matching API",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "earrings_available": len(matcher.earrings)
    }


@app.post("/recommend")
async def recommend_earrings(
    file: UploadFile = File(...),
    top_k: int = Query(
        default=3,
        ge=1,
        le=15,
        description="Number of earrings to return"
    )
):
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file."
        )

    try:
        # Read uploaded image
        contents = await file.read()

        necklace_image = Image.open(
            BytesIO(contents)
        ).convert("RGB")

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Unable to read the uploaded image."
        )

    try:
        recommendations = matcher.recommend(
            necklace_image=necklace_image,
            top_k=top_k
        )

        return {
            "input_image": file.filename,
            "recommendations": recommendations
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Matching failed: {str(e)}"
        )
from fastapi import FastAPI
import requests
from io import BytesIO
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim

from pydantic import BaseModel

app = FastAPI()

class DataClass(BaseModel):
    source_url: str
    generated_url: str
    



def fetch_image(url: str) -> Image.Image:
    """Fetch an image from a URL and return a PIL Image."""
    response = requests.get(url)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))

def resize_image(image: Image.Image, size: tuple) -> Image.Image:
    """Resize an image to the specified size."""
    return image.resize(size, Image.Resampling.LANCZOS)

def calculate_ssim(url1: str, url2: str) -> float:
    """Calculate SSIM between two images fetched from URLs."""
    # Fetch images
    img1 = fetch_image(url1)
    img2 = fetch_image(url2)

    # Convert to grayscale
    img1 = img1.convert('L')
    img2 = img2.convert('L')

    # Resize to the same dimensions
    size = (img1.width, img1.height)
    img2 = resize_image(img2, size)

    # Convert images to numpy arrays
    arr1 = np.array(img1)
    arr2 = np.array(img2)

    # Calculate SSIM
    score, _ = ssim(arr1, arr2, full=True)
    return score

def get_ssim_score(source_url, generated_url):
  ssim_score = calculate_ssim(source_url, generated_url)

  return ssim_score


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/get_ssim")
async def get_ssim(data: DataClass):
    ssim_score = get_ssim_score(data.source_url, data.generated_url)

    return {"score": ssim_score}



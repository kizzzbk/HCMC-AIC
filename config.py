import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
FAISS_DIR = BASE_DIR / "Faiss"
DATA_DIR = BASE_DIR / "data"
VECTORDB2_DIR = BASE_DIR / "vectordb2_output"

# FAISS Indices
FAISS_SIGLIP_PATH = str(FAISS_DIR / "faiss_siglip2.bin")
FAISS_OPENCLIP_PATH = str(FAISS_DIR / "faiss_openclip.bin")
FAISS_CAPTION_PATH = str(FAISS_DIR / "faiss_caption.bin")

# Database & Metadata Paths
METADATA_DB_PATH = str(BASE_DIR / "metadata.db")
METADATA_JSON_PATH = str(VECTORDB2_DIR / "vectordb2_openclip_metadata.json")
KEYFRAMES_DIR = os.getenv("KEYFRAMES_DIR", str(BASE_DIR / "keyframes"))
STATIC_DIR = str(BASE_DIR / "static")

# Keyframe Roots for Datasets (Supports single paths or lists of candidate paths)
KEYFRAME_ROOTS = {
    "L21": [BASE_DIR / "L21" / "output" / "keyframes", BASE_DIR / "L21" / "keyframes"],
    "L22": [BASE_DIR / "L22" / "output" / "keyframes", BASE_DIR / "L22" / "keyframes"],
    "L23": [BASE_DIR / "L23" / "output" / "keyframes", BASE_DIR / "L23" / "keyframes"],
    "L24": [BASE_DIR / "L24" / "output" / "keyframes", BASE_DIR / "L24" / "keyframes"],
    "L25": [
        BASE_DIR / "L25.1" / "keyframes",
        BASE_DIR / "L25.2" / "keyframes",
        BASE_DIR / "L25.3" / "keyframes",
        BASE_DIR / "L25" / "output" / "keyframes",
        BASE_DIR / "L25" / "keyframes",
    ],
    "L26": [
        BASE_DIR / "L26" / "output" / "keyframes",
        BASE_DIR / "L26" / "keyframes",
        BASE_DIR / "L26.1" / "keyframes",
        BASE_DIR / "L26.2" / "keyframes",
        BASE_DIR / "L26.3" / "keyframes",
        BASE_DIR / "L26_a.1" / "keyframes",
        BASE_DIR / "L26_a.2" / "keyframes",
        BASE_DIR / "L26_a.3" / "keyframes",
    ],
    "L27": [BASE_DIR / "L27" / "keyframes", BASE_DIR / "L27" / "output" / "keyframes"],
    "L28": [BASE_DIR / "L28" / "keyframes", BASE_DIR / "L28" / "output" / "keyframes"],
    "L29": [BASE_DIR / "L29" / "keyframes", BASE_DIR / "L29" / "output" / "keyframes"],
    "L30": [BASE_DIR / "L30" / "keyframes", BASE_DIR / "L30" / "output" / "keyframes"],
}

# Model Names / Configurations (Exact from HCMC_AI_Challenge.ipynb)
OPENCLIP_MODEL_NAME = os.getenv("OPENCLIP_MODEL_NAME", "ViT-B-32")
OPENCLIP_PRETRAINED = os.getenv("OPENCLIP_PRETRAINED", "laion2b_s34b_b79k")
SIGLIP_MODEL_NAME = os.getenv("SIGLIP_MODEL_NAME", "google/siglip2-base-patch16-naflex")
CAPTION_MODEL_NAME = os.getenv("CAPTION_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Retrieval & Fusion Parameters
DEFAULT_TOP_N = 100  # Number of candidates retrieved per FAISS index
DEFAULT_TOP_K = 20   # Number of results returned to user
RRF_K = 60           # RRF constant
WEIGHT_SIGLIP = 1.0
WEIGHT_OPENCLIP = 1.0
WEIGHT_CAPTION = 0.8

# Server Configuration
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))

# Submission Settings
SUBMISSION_URL = os.getenv("SUBMISSION_URL", "http://localhost:8000/api/mock_btc_submit")
SESSION_ID = os.getenv("SESSION_ID", "HCMC_AI_TEAM_01")

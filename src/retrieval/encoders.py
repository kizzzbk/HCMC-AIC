import torch
import numpy as np
import open_clip
from typing import Optional, Dict, Any
from transformers import AutoProcessor, AutoModel

from config import (
    OPENCLIP_MODEL_NAME,
    OPENCLIP_PRETRAINED,
    SIGLIP_MODEL_NAME,
    CAPTION_MODEL_NAME,
)

class OpenCLIPEncoder:
    """OpenCLIP text encoder — dùng đúng model/pretrained khi build FAISS index."""
    def __init__(self, model_name: str = OPENCLIP_MODEL_NAME, pretrained: str = OPENCLIP_PRETRAINED):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Encoder] Loading OpenCLIP model: {model_name} pretrained={pretrained} on {self.device}...")
        self.model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        print("[Encoder] OpenCLIP loaded successfully.")

    def encode_text(self, text: str) -> np.ndarray:
        tokens = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)  # normalize giống luc build index
            return emb.cpu().numpy().astype("float32")


class SigLIPEncoder:
    """Exact SigLIP2 implementation from Download_Keyframe.ipynb (Cell 9 & 12)"""
    def __init__(self, model_name: str = SIGLIP_MODEL_NAME):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Encoder] Loading SigLIP2 model: {model_name} on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print("[Encoder] SigLIP2 loaded successfully.")

    def encode_text(self, text: str) -> np.ndarray:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            # Dùng get_text_features() — có projection layer, khớp với get_image_features() khi build FAISS
            text_features = self.model.get_text_features(**inputs)
            # Normalize L2 — giống code embed ảnh: features / features.norm(p=2, dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features.cpu().numpy().astype("float32")


class CaptionEncoder:
    def __init__(self, model_name: str = CAPTION_MODEL_NAME):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Encoder] Loading Caption SentenceTransformer model: {model_name} on {self.device}...")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model.eval()
        print("[Encoder] Caption model loaded successfully.")

    def encode_text(self, text: str) -> np.ndarray:
        with torch.no_grad():
            emb = self.model.encode([text], normalize_embeddings=True)
            return emb.astype("float32")


import urllib.request
import urllib.parse
import json

def is_vietnamese(text: str) -> bool:
    vn_chars = set("àáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ")
    return any(c in vn_chars for c in text)

def translate_vi_to_en(text: str) -> str:
    if not text or not is_vietnamese(text):
        return text
    try:
        url = "https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=en&dt=t&q=" + urllib.parse.quote(text)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=3.5) as response:
            res = json.loads(response.read().decode("utf-8"))
            translated = "".join([part[0] for part in res[0] if part[0]])
            if translated.strip():
                return translated.strip()
    except Exception as e:
        print(f"[Encoder] Translation notice: using original query ({e})")
    return text


class ModelManager:
    _instance: Optional["ModelManager"] = None

    def __init__(self):
        self.openclip_encoder = OpenCLIPEncoder()
        self.siglip_encoder = SigLIPEncoder()
        try:
            self.caption_encoder = CaptionEncoder()
        except Exception as e:
            print(f"[Encoder] Warning: Failed to load Caption encoder ({e}). Will run in fallback mode.")
            self.caption_encoder = None

    @classmethod
    def get_instance(cls) -> "ModelManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def encode_query(self, query: str) -> Dict[str, Optional[np.ndarray]]:
        query_en = translate_vi_to_en(query)
        if query_en != query:
            try:
                print(f"[Encoder] Auto-translated query to EN: '{query_en}'")
            except Exception:
                pass

        siglip_vec = self.siglip_encoder.encode_text(query_en)
        openclip_vec = self.openclip_encoder.encode_text(query_en)
        caption_vec = self.caption_encoder.encode_text(query) if self.caption_encoder else None
        return {
            "siglip": siglip_vec,
            "openclip": openclip_vec,
            "caption": caption_vec,
        }

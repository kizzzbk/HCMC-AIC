import torch
import numpy as np
from typing import Optional, Dict, Any
from transformers import AutoProcessor, AutoModel
from sentence_transformers import SentenceTransformer

from config import (
    OPENCLIP_MODEL_NAME,
    SIGLIP_MODEL_NAME,
    CAPTION_MODEL_NAME,
)

class OpenCLIPEncoder:
    """Exact OpenCLIP implementation from Download_Keyframe.ipynb (Cell 9 & 12)"""
    def __init__(self, model_name: str = OPENCLIP_MODEL_NAME):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Encoder] Loading OpenCLIP model: {model_name} on {self.device}...")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model.eval()
        print("[Encoder] OpenCLIP loaded successfully.")

    def encode_text(self, text: str) -> np.ndarray:
        with torch.no_grad():
            emb = self.model.encode(text, convert_to_numpy=True).astype("float32")
            if emb.ndim == 1:
                emb = emb.reshape(1, -1)
            return emb


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
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        text_inputs = {k: v for k, v in inputs.items() if k in ['input_ids', 'attention_mask']}
        with torch.no_grad():
            text_encoder_output = self.model.text_model(**text_inputs)
            text_features = text_encoder_output.pooler_output
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

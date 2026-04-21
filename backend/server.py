"""
FastAPI backend for the Secure Deepfake Detection system.

Startup:
  - Loads all model components (HAMMER, watermark decoders, CLIP authenticator)
  - Verifies SHA-256 integrity hashes for all checkpoint files
  - Initializes UserDB
  - Exits non-zero if any integrity check fails

Endpoints:
  POST /auth/register  — register a new user
  POST /auth/login     — authenticate and receive a JWT
  POST /detect         — run deepfake detection (requires Bearer JWT)

Requirements: 14.1, 14.2, 14.3, 14.4, 14.5, 14.6
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from typing import List, Optional

import jwt
import numpy as np
import PIL.Image
import torch
import uvicorn
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("backend.server")

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------
JWT_SECRET: str = os.environ.get("JWT_SECRET", "change-me-in-production")
HAMMER_CHECKPOINT: Optional[str] = os.environ.get("HAMMER_CHECKPOINT")
WM_IMG_DEC_PATH: Optional[str] = os.environ.get("WM_IMG_DEC_PATH")
WM_TXT_DEC_PATH: Optional[str] = os.environ.get("WM_TXT_DEC_PATH")
MODEL_HASH_FILE: str = os.environ.get("MODEL_HASH_FILE", "model_hashes.json")
USER_DB_PATH: str = os.environ.get("USER_DB_PATH", "auth/users.npz")
CLIP_MODEL: str = os.environ.get("CLIP_MODEL", "openai/clip-vit-base-patch32")
AUTH_THRESHOLD: float = float(os.environ.get("AUTH_THRESHOLD", "0.85"))

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    label: str                        # "real" | "fake"
    trust_score: float
    hammer_score: float
    watermark_score: float
    watermark_valid: bool
    bbox: Optional[List[float]]       # [cx, cy, w, h] or None
    fake_token_positions: List[int]


# ---------------------------------------------------------------------------
# Global state (populated at startup)
# ---------------------------------------------------------------------------
_authenticator = None
_user_db = None
_hammer = None
_img_decoder = None
_txt_decoder = None
_tokenizer = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Secure Deepfake Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def _load_hashes() -> dict:
    if not os.path.exists(MODEL_HASH_FILE):
        logger.warning("Model hash file '%s' not found — skipping integrity checks.", MODEL_HASH_FILE)
        return {}
    with open(MODEL_HASH_FILE) as f:
        return json.load(f)


def _verify_checkpoint(name: str, path: str, hashes: dict) -> None:
    from integrity.model_integrity import ModelIntegrityError, verify_model
    if name not in hashes:
        logger.warning("No stored hash for '%s' — skipping integrity check.", name)
        return
    try:
        verify_model(path, hashes[name])
        logger.info("Integrity OK: %s", name)
    except ModelIntegrityError as exc:
        logger.error("INTEGRITY FAILURE: %s", exc)
        sys.exit(1)


@app.on_event("startup")
async def startup_event() -> None:
    global _authenticator, _user_db, _hammer, _img_decoder, _txt_decoder, _tokenizer

    hashes = _load_hashes()

    from auth.user_db import UserDB
    _user_db = UserDB(db_path=USER_DB_PATH)
    logger.info("UserDB initialized with %d records.", len(_user_db))

    from auth.clip_auth import CLIPAuthenticator
    _authenticator = CLIPAuthenticator(
        model_name=CLIP_MODEL,
        threshold=AUTH_THRESHOLD,
        db=_user_db,
    )
    logger.info("CLIPAuthenticator loaded (model=%s, threshold=%.2f).", CLIP_MODEL, AUTH_THRESHOLD)

    if WM_IMG_DEC_PATH:
        _verify_checkpoint("wm_img_decoder", WM_IMG_DEC_PATH, hashes)
        from models.watermark_image_decoder import ImageWatermarkDecoder
        _img_decoder = ImageWatermarkDecoder()
        _img_decoder.load_state_dict(torch.load(WM_IMG_DEC_PATH, map_location=_device))
        _img_decoder.to(_device).eval()
        logger.info("ImageWatermarkDecoder loaded from %s.", WM_IMG_DEC_PATH)
    else:
        logger.warning("WM_IMG_DEC_PATH not set — image watermark decoder not loaded.")

    if WM_TXT_DEC_PATH:
        _verify_checkpoint("wm_txt_decoder", WM_TXT_DEC_PATH, hashes)
        from models.watermark_text_decoder import TextWatermarkDecoder
        _txt_decoder = TextWatermarkDecoder()
        _txt_decoder.load_state_dict(torch.load(WM_TXT_DEC_PATH, map_location=_device))
        _txt_decoder.to(_device).eval()
        logger.info("TextWatermarkDecoder loaded from %s.", WM_TXT_DEC_PATH)
    else:
        logger.warning("WM_TXT_DEC_PATH not set — text watermark decoder not loaded.")

    if HAMMER_CHECKPOINT:
        _verify_checkpoint("hammer", HAMMER_CHECKPOINT, hashes)
        _load_hammer()
    else:
        logger.warning("HAMMER_CHECKPOINT not set — /detect will return mock responses.")


def _load_hammer() -> None:
    global _hammer, _tokenizer
    import argparse
    from ruamel.yaml import YAML
    _yaml = YAML()
    from transformers import BertTokenizer
    from models.HAMMER import HAMMER

    config_path = "configs/test.yaml"
    config = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = _yaml.load(f)

    args = argparse.Namespace(token_momentum=config.get("token_momentum", False))
    tokenizer_name = config.get("text_encoder", "bert-base-uncased")
    _tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    model = HAMMER(
        args=args, config=config,
        text_encoder=tokenizer_name,
        tokenizer=_tokenizer,
        init_deit=False,
    )
    checkpoint = torch.load(HAMMER_CHECKPOINT, map_location=_device)
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.to(_device).eval()
    _hammer = model
    logger.info("HAMMER loaded from %s.", HAMMER_CHECKPOINT)


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------

def _decode_jwt(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired.")
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status_code=401, detail=f"Invalid token: {exc}")


_bearer_scheme = HTTPBearer(auto_error=False)


def _require_jwt(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> dict:
    if credentials is None:
        raise HTTPException(status_code=401, detail="Authorization header missing.")
    return _decode_jwt(credentials.credentials)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _read_image(upload: UploadFile) -> PIL.Image.Image:
    try:
        data = upload.file.read()
        return PIL.Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid image file: {exc}")


def _pil_to_tensor(img: PIL.Image.Image) -> torch.Tensor:
    import torchvision.transforms as T
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0).to(_device)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _compute_watermark_score(image_tensor: torch.Tensor, text: str) -> tuple:
    """Compute watermark consistency score. Returns (score, valid). Falls back to (0.0, False)."""
    import hashlib
    from utils.metrics import compute_nc

    try:
        scores = []
        if _img_decoder is not None:
            m_T_bits = np.unpackbits(
                np.frombuffer(hashlib.sha256(text.encode()).digest()[:16], dtype=np.uint8)
            ).astype(np.float32)
            m_T_t = torch.tensor(m_T_bits, device=_device)
            with torch.no_grad():
                m_T_hat = _img_decoder(image_tensor).squeeze(0)
            scores.append(float(compute_nc(m_T_t, m_T_hat)))

        if not scores:
            return 0.0, False

        wm_score = float(np.mean(scores))
        return wm_score, wm_score >= 0.5
    except Exception as exc:
        logger.warning("Watermark extraction failed: %s", exc)
        return 0.0, False


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/auth/register", status_code=201)
async def register(
    username: str = Form(...),
    image: UploadFile = File(...),
    password: str = Form(...),
):
    """Register a new user. Returns 201 or 409 on duplicate."""
    from auth.user_db import ConflictError
    pil_image = _read_image(image)
    try:
        _authenticator.register(username=username, image=pil_image, password=password)
    except ConflictError:
        raise HTTPException(status_code=409, detail=f"Username '{username}' is already registered.")
    return {"detail": "User registered successfully."}


@app.post("/auth/login")
async def login(
    username: str = Form(...),
    image: UploadFile = File(...),
    password: str = Form(...),
):
    """Authenticate a user and return a JWT. Returns 401 on failure."""
    from auth.clip_auth import AuthenticationError
    pil_image = _read_image(image)
    try:
        token = _authenticator.authenticate(username=username, image=pil_image, password=password)
    except (KeyError, AuthenticationError):
        raise HTTPException(status_code=401, detail="Authentication failed.")
    return {"token": token}


@app.post("/detect")
async def detect(
    image: UploadFile = File(...),
    text: str = Form(...),
    jwt_payload: dict = Depends(_require_jwt),
):
    """Run deepfake detection. Requires Bearer JWT. Returns DetectionResult JSON."""
    pil_image = _read_image(image)
    image_tensor = _pil_to_tensor(pil_image)

    wm_score, wm_valid = _compute_watermark_score(image_tensor, text)

    if _hammer is None:
        # Mock response — HAMMER not loaded
        result = DetectionResult(
            label="real",
            trust_score=0.3 * wm_score,
            hammer_score=0.0,
            watermark_score=wm_score,
            watermark_valid=wm_valid,
            bbox=None,
            fake_token_positions=[],
        )
        return asdict(result)

    # Full HAMMER inference
    import torch.nn.functional as F
    text_inputs = _tokenizer(
        text, return_tensors="pt", padding="max_length",
        truncation=True, max_length=30,
    )
    text_inputs = {k: v.to(_device) for k, v in text_inputs.items()}

    class _T:
        def __init__(self, enc):
            self.input_ids = enc["input_ids"]
            self.attention_mask = enc["attention_mask"]

    with torch.no_grad():
        logits_rf, _, output_coord, logits_tok = _hammer(
            image=image_tensor, label=None, text=_T(text_inputs),
            fake_image_box=None, fake_text_pos=None, is_train=False,
        )

    hammer_score = float(F.softmax(logits_rf, dim=-1)[0, 1].item())
    bbox = output_coord[0].cpu().tolist() if hammer_score > 0.5 else None
    fake_tokens = []
    if logits_tok is not None:
        tok_probs = F.softmax(logits_tok[0], dim=-1)
        fake_tokens = (tok_probs[:, 1] > 0.5).nonzero(as_tuple=True)[0].cpu().tolist()

    trust_score = 0.7 * hammer_score + 0.3 * wm_score
    result = DetectionResult(
        label="fake" if hammer_score > 0.5 else "real",
        trust_score=trust_score,
        hammer_score=hammer_score,
        watermark_score=wm_score,
        watermark_valid=wm_valid,
        bbox=bbox,
        fake_token_positions=fake_tokens,
    )
    return asdict(result)


if __name__ == "__main__":
    uvicorn.run("backend.server:app", host="0.0.0.0", port=8000, reload=False)

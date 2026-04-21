# Implementation Plan: Secure Deepfake Detection System

## Overview

Incrementally extend the existing DGM4 + HAMMER repository with watermarking modules, CLIP-based
authentication, model integrity verification, a FastAPI backend, and a unified Next.js frontend.
Each task builds on the previous one; no code is left unintegrated.

## Tasks

- [x] 1. Create metrics module (`utils/metrics.py`)
  - Create `utils/` directory and `utils/__init__.py`
  - Implement `compute_psnr`, `compute_nc` with `ValueError` on shape mismatch
  - Implement `compute_auc`, `compute_eer`, `compute_acc` wrapping existing sklearn calls from `train.py`
  - Implement `compute_iou_mean`, `compute_iou_at_50`, `compute_iou_at_75`, `compute_iou_at_95`
  - Implement `compute_token_f1`, `compute_token_precision`, `compute_token_recall`
  - Implement `compute_map` delegating to `AveragePrecisionMeter.value().mean()`
  - `compute_psnr(T, T)` must return `float('inf')`; `compute_nc(T, T)` must return `1.0`
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 16.3, 16.4_

  - [ ]* 1.1 Write property test for PSNR identity and correctness (Property 5)
    - **Property 5: PSNR identity and correctness**
    - **Validates: Requirements 8.1, 8.4, 16.3**
    - Use `hypothesis` with `@given(st.integers(1,4), st.integers(1,256))` for shape
    - Verify `compute_psnr(T, T)` returns `float('inf')` or > 100 dB
    - Verify formula `10 * log10(1.0 / MSE)` for known MSE

  - [ ]* 1.2 Write property test for NC identity, range, and correctness (Property 6)
    - **Property 6: NC identity, range, and correctness**
    - **Validates: Requirements 8.2, 8.5, 16.4**
    - Verify `compute_nc(T, T) == 1.0`
    - Verify `compute_nc(T, -T) == -1.0`
    - Verify result is always in `[-1.0, 1.0]`

  - [ ]* 1.3 Write property test for metrics ValueError on shape mismatch (Property 7)
    - **Property 7: Metrics ValueError on shape mismatch**
    - **Validates: Requirements 8.6**
    - Generate two tensors with different shapes; assert `ValueError` is raised for both `compute_psnr` and `compute_nc`

- [x] 2. Implement image watermark encoder (`models/watermark_image_encoder.py`)
  - ResNet-18 encoder path (pretrained=False) + 4-layer transposed-conv decoder with skip connections
  - Linear projection of 128-bit watermark to spatial feature map, concatenated at bottleneck
  - `forward(image, m_T) -> I_w` using `I_w = image + alpha * f_theta(image, m_T)`
  - `alpha` is a learnable scalar initialized to 0.03, clamped to [0.01, 0.05] in `forward`
  - Output shape must equal input shape `(B, 3, 224, 224)`
  - _Requirements: 2.1, 2.2, 2.3_

  - [x]* 2.1 Write property test for image encoder output shape preservation (Property 1)
    - **Property 1: Image encoder output shape preservation**
    - **Validates: Requirements 2.2, 2.3**
    - `@given(batch_size=st.integers(1, 4))` — assert output shape equals `(B, 3, 224, 224)`

- [x] 3. Implement image watermark decoder (`models/watermark_image_decoder.py`)
  - 4-layer strided conv → GlobalAvgPool → `Linear(512, 128)` → Sigmoid
  - `forward(image_w) -> m_T_hat` with output shape `(B, 128)`, values in `[0, 1]`
  - Raise `ValueError` on unexpected input shape
  - _Requirements: 3.1_

- [x] 4. Implement text watermark encoder (`models/watermark_text_encoder.py`)
  - `P: Linear(128, 768)` learned projection; `alpha` clamped to [0.01, 0.05]
  - `forward(embeddings, m_I) -> E_w` using `E_w = embeddings + alpha * P(m_I).unsqueeze(1).expand_as(embeddings)`
  - Output shape must equal input shape `(B, seq_len, hidden_dim)`
  - _Requirements: 4.1, 4.2, 4.3_

  - [ ]* 4.1 Write property test for text encoder output shape preservation (Property 2)
    - **Property 2: Text encoder output shape preservation**
    - **Validates: Requirements 4.2, 4.3**
    - `@given(batch_size=st.integers(1,4), seq_len=st.integers(1,128))` — assert output shape equals input shape

- [x] 5. Implement text watermark decoder (`models/watermark_text_decoder.py`)
  - MeanPool over `seq_len` → `Linear(768, 256)` → GELU → `Linear(256, 128)` → Sigmoid
  - `forward(embeddings_w) -> m_I_hat` with output shape `(B, 128)`, values in `[0, 1]`
  - Raise `ValueError` on unexpected input shape
  - _Requirements: 5.1_

- [x] 6. Checkpoint — Ensure all watermark module unit tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Implement model integrity module (`integrity/model_integrity.py`)
  - Create `integrity/` directory and `integrity/__init__.py`
  - Implement `compute_file_hash(path) -> str` using SHA-256 with streaming reads
  - Implement `save_hashes(model_paths, hash_file)` writing JSON
  - Implement `verify_model(path, expected_hash)` raising `ModelIntegrityError` on mismatch
  - _Requirements: 13.1, 13.2, 13.3, 13.4_

  - [ ]* 7.1 Write property test for model integrity hash verification (Property 14)
    - **Property 14: Model integrity hash verification**
    - **Validates: Requirements 13.2, 13.3**
    - Write a temp file, save hash, flip one byte, assert `ModelIntegrityError` is raised
    - Verify unmodified file passes without raising

- [x] 8. Implement user DB (`auth/user_db.py`)
  - Create `auth/` directory and `auth/__init__.py`
  - `UserDB.__init__` loads `.npz` on startup; initializes empty store + logs warning on any error
  - `save(username, embedding)` raises `ConflictError` if username exists
  - `lookup(username)` returns embedding or raises `KeyError`
  - `persist()` writes atomically to `db_path`
  - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5_

  - [ ]* 8.1 Write property test for username conflict detection (Property 13)
    - **Property 13: Username conflict detection**
    - **Validates: Requirements 12.4**
    - Register a username, attempt to register again, assert `ConflictError` and original embedding unchanged

- [x] 9. Implement CLIP authenticator (`auth/clip_auth.py`)
  - Load frozen `openai/clip-vit-base-patch32`; weights never updated
  - `compute_embedding(image, password) -> np.ndarray` — `normalize(CLIP_image + CLIP_text)`, shape `(512,)`
  - `register(username, image, password)` — delegates to `UserDB.save`; raises `ConflictError` if duplicate
  - `authenticate(username, image, password) -> str` — computes cosine similarity; returns JWT if > threshold; raises `AuthenticationError` otherwise
  - JWT signed with configurable secret and expiry
  - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7_

  - [ ]* 9.1 Write property test for user embedding unit normalization (Property 11)
    - **Property 11: User embedding is unit-normalized**
    - **Validates: Requirements 11.2**
    - `@given` random image arrays and password strings; assert `||e||_2 ≈ 1.0` within 1e-5

  - [ ]* 9.2 Write property test for authentication threshold decision (Property 12)
    - **Property 12: Authentication threshold decision**
    - **Validates: Requirements 11.4, 11.5**
    - Mock stored embedding; verify grant when similarity > threshold, deny when ≤ threshold

- [x] 10. Update `configs/train.yaml` with watermark training additions
  - Add `watermark_alpha: 0.03`, `watermark_dim: 128`, `subset_size: 35000`
  - Override `image_res: 224`, `use_fp16: true`, `freeze_encoders: true`, `epochs: 8`
  - Add `loss_watermark_wgt: 1.0`
  - _Requirements: 1.2, 1.3, 1.6_

- [x] 11. Modify training pipeline (`train.py`) — balanced subset, frozen encoders, FP16, joint loss
  - Add `create_balanced_subset(dataset, subset_size)` that returns equal real/fake samples
  - Add `freeze_encoders(model)` that sets `requires_grad=False` on `visual_encoder` and `text_encoder`
  - Wrap forward + backward in `torch.cuda.amp.autocast`; use `GradScaler` for optimizer step
  - Per batch: compute `m_T = SHA256(text)[:16]` and `m_I = SHA256(image_features)[:16]` as float32 tensors
  - Encode: `I_w = ImageWatermarkEncoder(I, m_T)`, `E_w = TextWatermarkEncoder(E, m_I)`
  - Decode: `m_T_hat = ImageWatermarkDecoder(I_w)`, `m_I_hat = TextWatermarkDecoder(E_w)`
  - Compute `L_image = MSE(I, I_w) + BCE(m_T, m_T_hat)` and `L_text = (1 - cosine_sim(E, E_w)) + BCE(m_I, m_I_hat)`
  - Compute `L_watermark = L_image + L_text`; pass `I_w`, `E_w` to HAMMER for `L_classification`
  - Compute `L_final = L_watermark + L_classification`; single backward pass
  - Log `L_watermark`, `L_classification`, `L_final`, PSNR, NC to TensorBoard each step
  - Apply Gaussian noise (σ ∈ [0.0, 0.05]) and JPEG compression simulation (quality ∈ [70, 95]) to `I_w` during training
  - Log GradScaler skip warnings when FP16 overflow occurs
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 7.1, 7.2, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8_

  - [ ]* 11.1 Write property test for balanced sampler equal class counts (Property 15)
    - **Property 15: Balanced sampler equal class counts**
    - **Validates: Requirements 1.1**
    - `@given(n_real=st.integers(1,1000), n_fake=st.integers(1,1000))` — assert subset size is `2 * min(N, M)` with equal class counts

  - [ ]* 11.2 Write property test for encoder freeze preserves trainability (Property 16)
    - **Property 16: Encoder freeze preserves trainability of other parameters**
    - **Validates: Requirements 1.4, 1.5**
    - After `freeze_encoders(model)`, assert all `visual_encoder` and `text_encoder` params have `requires_grad=False`
    - Assert fusion layers, classification head, and watermark module params have `requires_grad=True`

- [x] 12. Checkpoint — Ensure training pipeline tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 13. Modify inference pipeline (`test.py`) — trust score and watermark consistency
  - Load `ImageWatermarkDecoder` and `TextWatermarkDecoder` alongside HAMMER
  - Verify all model files via `integrity/model_integrity.py` before loading weights
  - Per sample: extract `m_T_hat = ImageWatermarkDecoder(I)`, `m_I_hat = TextWatermarkDecoder(E)`
  - Compute `watermark_score = mean(NC(m_T_hat, expected_m_T), NC(m_I_hat, expected_m_I))`
  - Wrap watermark extraction in try/except; on any error set `watermark_score = 0.0`, `watermark_valid = False`
  - Set `watermark_valid = watermark_score >= 0.5`
  - Compute `trust_score = 0.7 * hammer_score + 0.3 * watermark_score`
  - Output `DetectionResult` dataclass per sample with all required fields
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 13.2, 13.3_

  - [ ]* 13.1 Write property test for trust score formula correctness (Property 8)
    - **Property 8: Trust score formula correctness**
    - **Validates: Requirements 10.4**
    - `@given(hammer=st.floats(0,1), wm=st.floats(0,1))` — assert `trust_score == 0.7*hammer + 0.3*wm`

  - [ ]* 13.2 Write property test for watermark score defaults to 0.0 on invalid input (Property 9)
    - **Property 9: Watermark score defaults to 0.0 on invalid input**
    - **Validates: Requirements 10.6**
    - Pass `None`, wrong-shape tensors, and NaN-filled tensors; assert `watermark_score == 0.0` and no unhandled exception

  - [ ]* 13.3 Write property test for consistency flag threshold (Property 10)
    - **Property 10: Consistency flag threshold**
    - **Validates: Requirements 6.3**
    - `@given(score=st.floats(0,1))` — assert `watermark_valid == (score >= 0.5)`

- [x] 14. Implement Python FastAPI backend (`backend/server.py`)
  - Create `backend/` directory
  - On startup: load all five model components, verify integrity hashes, initialize `UserDB`; exit non-zero if any hash fails
  - `POST /auth/login` — accepts multipart `{image: File, password: str}`; calls `CLIPAuthenticator.authenticate`; returns `{token}` or 401
  - `POST /detect` — accepts multipart `{image: File, text: str}` + `Authorization: Bearer <token>`; validates JWT; runs inference pipeline; returns `DetectionResult` JSON
  - Return 422 on invalid image (PIL.Image.open fails), 401 on missing/expired JWT, 409 on duplicate registration
  - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5, 14.6_

- [x] 15. Scaffold Next.js application (`web/`)
  - Initialize Next.js App Router project in `web/` with TypeScript and Tailwind CSS
  - Add `fast-check` as a dev dependency for TypeScript property tests
  - Create `web/package.json` with all required dependencies
  - _Requirements: 14.7, 15.8_

- [x] 16. Implement Next.js API routes
  - `web/app/api/login/route.ts` — `POST /api/login`: accepts multipart `{image, password}`; proxies to Python backend `/auth/login`; returns `{token}` or 401
  - `web/app/api/predict/route.ts` — `POST /api/predict`: validates `Authorization: Bearer <token>`; proxies to Python backend `/detect`; returns `DetectionResult` or 401/422
  - Define `DetectionResult` TypeScript interface in `web/types/detection.ts`
  - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5_

- [x] 17. Implement Next.js frontend pages and components
  - `web/components/LoginForm.tsx` — image upload + password field + submit; calls `POST /api/login`; stores token in `sessionStorage`
  - `web/components/DetectionForm.tsx` — image upload + text field + submit; calls `POST /api/predict` with Bearer token; shows loading indicator
  - `web/components/ResultVisualization.tsx` — displays label, trust score, watermark validity; overlays bounding box on image canvas; highlights fake token positions in text
  - `web/app/page.tsx` — renders `LoginForm`; redirects to `/detect` on success
  - `web/app/detect/page.tsx` — guards route (redirect to `/` if no token); renders `DetectionForm` + `ResultVisualization`
  - Redirect to login page when session token is absent or expired (401 response)
  - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.5, 15.6, 15.7_

- [x] 18. Checkpoint — Ensure all tests pass and components are wired together
  - Ensure all tests pass, ask the user if questions arise.

- [x] 19. Update `requirements.txt`
  - Add `fastapi`, `uvicorn[standard]`, `python-multipart`, `python-jose[cryptography]`
  - Add `transformers>=4.30`, `open-clip-torch` or `clip` (for CLIP authenticator)
  - Add `hypothesis` for property-based tests
  - Add `Pillow`, `scipy`, `scikit-learn` if not already present
  - _Requirements: all_

- [x] 20. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Property tests use `hypothesis` (Python) and `fast-check` (TypeScript)
- Each property test references the design document property number and the requirement clause it validates
- Checkpoints ensure incremental validation before moving to the next layer
- The Python FastAPI backend is an implementation detail managed by Next.js; it is not a separately deployed service visible to end users

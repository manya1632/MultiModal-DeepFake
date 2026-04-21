# Requirements Document

## Introduction

This feature transforms the existing DGM4 + HAMMER multimodal deepfake detection repository into a
lightweight, secure, production-ready system. The enhanced system adds cross-modal neural watermarking
(image and text), CLIP-based access control (image + password authentication), watermark quality
metrics (PSNR, NC), a unified Next.js application (API routes + frontend), and a React-based web
interface. All new components are designed
to train within 6–8 hours on a single T4 GPU using FP16 mixed precision and a 30K–40K balanced
sample subset, while preserving the full detection and grounding capabilities of the original HAMMER
model.

---

## Glossary

- **System**: The complete secure multimodal deepfake detection system described in this document.
- **HAMMER**: The existing Hierarchical Multi-modal Manipulation rEasoning tRansformer model
  (`models/HAMMER.py`) used for deepfake detection and grounding.
- **Image_Watermark_Encoder**: The CNN-based encoder (`models/watermark_image_encoder.py`) that
  embeds a 128-bit watermark into an image.
- **Image_Watermark_Decoder**: The CNN-based decoder (`models/watermark_image_decoder.py`) that
  extracts a 128-bit watermark from a watermarked image.
- **Text_Watermark_Encoder**: The projection-layer encoder (`models/watermark_text_encoder.py`) that
  embeds a watermark into token embeddings.
- **Text_Watermark_Decoder**: The decoder (`models/watermark_text_decoder.py`) that extracts a
  watermark from watermarked token embeddings.
- **CLIP_Authenticator**: The CLIP-based access control module (`auth/clip_auth.py`) that registers
  and verifies users via image + password embeddings.
- **User_DB**: The secure user credential store (`auth/user_db.py`) that persists hashed user
  embeddings.
- **Metrics_Module**: The utility module (`utils/metrics.py`) that computes detection metrics (AUC, EER, ACC, IoU_mean, IoU@50, IoU@75, IoU@95, token-level F1/Precision/Recall, mAP) and watermark quality metrics (PSNR, NC).
- **API_Server**: The Next.js API routes (`/api/login` and `/api/predict`) that serve as the backend within the unified Next.js application.
- **Frontend**: The unified Next.js application that provides both the web interface and the API routes for authentication and result visualization.
- **Training_Pipeline**: The modified `train.py` that orchestrates watermark training alongside
  HAMMER classification training.
- **Inference_Pipeline**: The modified `test.py` that runs watermark extraction, consistency
  checking, and HAMMER detection to produce a final trust score.
- **Watermark**: A 128-bit binary vector derived from SHA-256 of the paired modality's content,
  used to verify cross-modal consistency.
- **Trust_Score**: The final output score computed as `0.7 * hammer_score + 0.3 * watermark_score`.
- **PSNR**: Peak Signal-to-Noise Ratio, a measure of watermark imperceptibility (target > 40 dB).
- **NC**: Normalized Correlation, a measure of watermark extraction accuracy (target > 0.95).
- **FP16**: 16-bit floating-point mixed precision training via `torch.cuda.amp`.
- **T4_GPU**: NVIDIA T4 GPU (16 GB VRAM), the target training hardware.
- **DGM4_Dataset**: The existing DGM4 dataset of 230K image-text pairs used as the data source.
- **Balanced_Subset**: A 30K–40K sample subset with equal real and fake samples drawn from
  DGM4_Dataset.

---

## Requirements

---

### Requirement 1: Lightweight Training Configuration

**User Story:** As a researcher, I want to train the full system on a single T4 GPU within 6–8 hours,
so that the system is accessible without expensive multi-GPU infrastructure.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL sample a Balanced_Subset of 30,000–40,000 image-text pairs from
   DGM4_Dataset with equal numbers of real and fake samples.
2. THE Training_Pipeline SHALL resize all input images to 224×224 pixels before feeding them to any
   model component.
3. THE Training_Pipeline SHALL apply FP16 mixed precision using `torch.cuda.amp.autocast` and
   `torch.cuda.amp.GradScaler` during all forward and backward passes.
4. THE Training_Pipeline SHALL freeze the weights of HAMMER's `visual_encoder` and `text_encoder`
   modules so that gradients are not computed for those parameters.
5. THE Training_Pipeline SHALL train only the fusion layers, classification head, and watermark
   encoder-decoder modules (Image_Watermark_Encoder, Image_Watermark_Decoder,
   Text_Watermark_Encoder, Text_Watermark_Decoder).
6. THE Training_Pipeline SHALL run for 6–8 epochs with a batch size of 32.
7. WHEN training completes, THE Training_Pipeline SHALL log total wall-clock time and confirm it
   does not exceed 8 hours on a T4 GPU.

---

### Requirement 2: Image Watermark Encoder

**User Story:** As a developer, I want to embed an imperceptible 128-bit watermark into an image,
so that the image carries a verifiable cross-modal signature.

#### Acceptance Criteria

1. THE Image_Watermark_Encoder SHALL accept an original image tensor `I` of shape `(B, 3, 224, 224)`
   and a 128-bit binary watermark vector `m_T` derived from `SHA256(text)`.
2. THE Image_Watermark_Encoder SHALL produce a watermarked image `I_w` using the embedding equation
   `I_w = I + α * fθ(I, m_T)`, where `fθ` is a UNet or ResNet-18 based CNN and `α` is a scalar in
   the range [0.01, 0.05].
3. THE Image_Watermark_Encoder SHALL output `I_w` with the same shape `(B, 3, 224, 224)` as the
   input image `I`.
4. WHEN the Image_Watermark_Encoder is trained, THE Image_Watermark_Encoder SHALL minimize the
   combined loss `L_image = MSE(I, I_w) + BCE(m_T, m_T_hat)` where `m_T_hat` is the decoded
   watermark from Image_Watermark_Decoder.
5. THE Image_Watermark_Encoder SHALL produce watermarked images with PSNR greater than 40 dB
   relative to the original image, as measured by Metrics_Module.

---

### Requirement 3: Image Watermark Decoder

**User Story:** As a developer, I want to extract the embedded watermark from a watermarked image,
so that cross-modal consistency can be verified at inference time.

#### Acceptance Criteria

1. THE Image_Watermark_Decoder SHALL accept a watermarked image tensor `I_w` of shape
   `(B, 3, 224, 224)` and output a predicted 128-bit watermark vector `m_T_hat`.
2. THE Image_Watermark_Decoder SHALL achieve a Normalized Correlation (NC) score greater than 0.95
   between the original watermark `m_T` and the extracted watermark `m_T_hat`, as measured by
   Metrics_Module.
3. WHEN the Image_Watermark_Decoder is applied to an unwatermarked image, THE
   Image_Watermark_Decoder SHALL return a watermark vector with NC less than 0.5 relative to any
   expected watermark, indicating absence of a valid watermark.

---

### Requirement 4: Text Watermark Encoder

**User Story:** As a developer, I want to embed a watermark into text token embeddings, so that the
text carries a verifiable cross-modal signature derived from the paired image.

#### Acceptance Criteria

1. THE Text_Watermark_Encoder SHALL accept token embeddings `E` of shape `(B, seq_len, hidden_dim)`
   and a watermark vector `m_I` derived from image features.
2. THE Text_Watermark_Encoder SHALL produce watermarked embeddings `E_w` using the equation
   `E_w = E + α * P(m_I)`, where `P` is a learned linear projection layer and `α` is in [0.01, 0.05].
3. THE Text_Watermark_Encoder SHALL output `E_w` with the same shape `(B, seq_len, hidden_dim)` as
   the input embeddings `E`.
4. WHEN the Text_Watermark_Encoder is trained, THE Text_Watermark_Encoder SHALL minimize the
   combined loss `L_text = (1 - cosine_similarity(E, E_w)) + BCE(m_I, m_I_hat)` where `m_I_hat` is
   the decoded watermark from Text_Watermark_Decoder.

---

### Requirement 5: Text Watermark Decoder

**User Story:** As a developer, I want to extract the embedded watermark from watermarked text
embeddings, so that cross-modal consistency can be verified at inference time.

#### Acceptance Criteria

1. THE Text_Watermark_Decoder SHALL accept watermarked token embeddings `E_w` of shape
   `(B, seq_len, hidden_dim)` and output a predicted 128-bit watermark vector `m_I_hat`.
2. THE Text_Watermark_Decoder SHALL achieve NC greater than 0.95 between the original watermark
   `m_I` and the extracted watermark `m_I_hat`, as measured by Metrics_Module.
3. WHEN the Text_Watermark_Decoder is applied to embeddings without an embedded watermark, THE
   Text_Watermark_Decoder SHALL return a watermark vector with NC less than 0.5 relative to any
   expected watermark.

---

### Requirement 6: Cross-Modal Watermark Consistency

**User Story:** As a security engineer, I want to verify that the watermarks embedded in an image
and its paired text are mutually consistent, so that cross-modal tampering can be detected.

#### Acceptance Criteria

1. THE Inference_Pipeline SHALL compute a cross-modal consistency score by checking that
   `D_I(I_w) ≈ SHA256(text)` and `D_T(E_w) ≈ hash(image_features)`, where `D_I` is
   Image_Watermark_Decoder and `D_T` is Text_Watermark_Decoder.
2. THE Inference_Pipeline SHALL compute the consistency score as the mean NC between the extracted
   watermarks and their expected hash values.
3. WHEN the consistency score is below 0.5, THE Inference_Pipeline SHALL flag the sample as having
   an invalid or missing watermark.
4. THE Inference_Pipeline SHALL include the consistency score as the `watermark_score` component of
   the Trust_Score computation.

---

### Requirement 7: Watermark Robustness and Imperceptibility

**User Story:** As a researcher, I want the watermark to survive common image degradations, so that
the system remains reliable under real-world conditions.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL apply Gaussian noise augmentation (σ ∈ [0.0, 0.05]) to watermarked
   images during training to improve robustness.
2. THE Training_Pipeline SHALL apply JPEG compression simulation (quality factor ∈ [70, 95]) to
   watermarked images during training to improve robustness.
3. THE Image_Watermark_Encoder SHALL constrain `α` to the range [0.01, 0.05] to maintain
   imperceptibility.
4. WHEN evaluated on the validation split, THE Image_Watermark_Encoder SHALL produce watermarked
   images with PSNR greater than 40 dB and the Image_Watermark_Decoder SHALL achieve NC greater
   than 0.95.

---

### Requirement 8: Watermark Quality Metrics

**User Story:** As a researcher, I want to compute standardized watermark quality metrics, so that I
can objectively evaluate imperceptibility and extraction accuracy.

#### Acceptance Criteria

1. THE Metrics_Module SHALL expose a `compute_psnr(original: Tensor, watermarked: Tensor) -> float`
   function that computes `PSNR = 10 * log10(MAX² / MSE(original, watermarked))` where `MAX = 1.0`
   for normalized images.
2. THE Metrics_Module SHALL expose a `compute_nc(m: Tensor, m_hat: Tensor) -> float` function that
   computes `NC = (Σ mᵢ * m̂ᵢ) / (sqrt(Σ mᵢ²) * sqrt(Σ m̂ᵢ²))`.
3. THE Metrics_Module SHALL expose all detection metrics from the DGM4/HAMMER paper:
   - `compute_auc` — Area Under the ROC Curve for binary classification
   - `compute_eer` — Equal Error Rate for binary classification
   - `compute_acc` — Accuracy for binary classification
   - `compute_iou_mean` — mean Intersection over Union across all IoU thresholds for image grounding
   - `compute_iou_at_50` — IoU@50 (IoU threshold 0.50) for image grounding
   - `compute_iou_at_75` — IoU@75 (IoU threshold 0.75) for image grounding
   - `compute_iou_at_95` — IoU@95 (IoU threshold 0.95) for image grounding
   - `compute_token_f1`, `compute_token_precision`, `compute_token_recall` — token-level F1, Precision, and Recall for text grounding
   - `compute_map` — mean Average Precision across IoU thresholds
4. WHEN `compute_psnr` receives two tensors of identical shape, THE Metrics_Module SHALL return a
   scalar float value in decibels.
5. WHEN `compute_nc` receives two tensors of identical shape, THE Metrics_Module SHALL return a
   scalar float value in the range [-1.0, 1.0].
6. IF `compute_psnr` or `compute_nc` receives tensors of mismatched shapes, THEN THE Metrics_Module
   SHALL raise a `ValueError` with a descriptive message.

---

### Requirement 9: Integrated Training Pipeline

**User Story:** As a developer, I want a single training script that jointly trains the watermark
modules and the HAMMER detection head, so that all components are optimized together.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL, for each training batch, compute `m_T = SHA256(text)` and
   `m_I = hash(image_features)` as 128-bit binary watermark vectors.
2. THE Training_Pipeline SHALL embed watermarks to produce `I_w = Image_Watermark_Encoder(I, m_T)`
   and `E_w = Text_Watermark_Encoder(E, m_I)`.
3. THE Training_Pipeline SHALL decode watermarks to produce `m_T_hat = Image_Watermark_Decoder(I_w)`
   and `m_I_hat = Text_Watermark_Decoder(E_w)`.
4. THE Training_Pipeline SHALL compute `L_watermark = L_image + L_text` as defined in Requirements
   2 and 4.
5. THE Training_Pipeline SHALL pass the watermarked inputs `I_w` and `E_w` to HAMMER and compute
   `L_classification` as the weighted sum of HAMMER's existing loss components (MAC, BIC, bbox,
   giou, TMG, MLC).
6. THE Training_Pipeline SHALL compute the final loss as
   `L_final = L_watermark + L_classification` and perform a single backward pass.
7. THE Training_Pipeline SHALL log `L_watermark`, `L_classification`, `L_final`, PSNR, and NC
   metrics to TensorBoard at each training step.
8. THE Training_Pipeline SHALL save the best checkpoint based on validation AUC_cls, consistent
   with the existing checkpoint saving logic.

---

### Requirement 10: Integrated Inference Pipeline

**User Story:** As a developer, I want a single inference script that produces a combined trust
score from both HAMMER detection and watermark consistency, so that results are more robust.

#### Acceptance Criteria

1. THE Inference_Pipeline SHALL extract watermarks from input image and text using
   Image_Watermark_Decoder and Text_Watermark_Decoder.
2. THE Inference_Pipeline SHALL compute the watermark consistency score as defined in Requirement 6.
3. THE Inference_Pipeline SHALL run HAMMER detection to obtain `hammer_score` (real/fake probability).
4. THE Inference_Pipeline SHALL compute
   `Trust_Score = 0.7 * hammer_score + 0.3 * watermark_score`.
5. THE Inference_Pipeline SHALL output, for each sample: the Fake/Real label, bounding box
   coordinates of manipulated image regions, highlighted fake token positions in text, watermark
   validity flag, and Trust_Score.
6. WHEN the watermark inputs are missing or invalid, THE Inference_Pipeline SHALL flag the sample
   and set `watermark_score = 0.0` rather than raising an unhandled exception.

---

### Requirement 11: CLIP-Based User Authentication

**User Story:** As a security engineer, I want users to authenticate using a profile image and a
text password via CLIP embeddings, so that access to the detection system is controlled without
storing raw passwords.

#### Acceptance Criteria

1. THE CLIP_Authenticator SHALL accept a profile image and a text password string as registration
   inputs.
2. THE CLIP_Authenticator SHALL compute a user embedding as
   `user_embedding = normalize(CLIP_image(image) + CLIP_text(password))` using a pre-trained CLIP
   model (e.g., `openai/clip-vit-base-patch32`).
3. THE CLIP_Authenticator SHALL store the user embedding in User_DB associated with a username,
   without storing the raw image or raw password.
4. WHEN a login request is received, THE CLIP_Authenticator SHALL compute the query embedding using
   the same normalization procedure and compare it to the stored embedding using cosine similarity.
5. WHEN the cosine similarity between the query embedding and the stored embedding exceeds a
   configurable threshold (default 0.85), THE CLIP_Authenticator SHALL grant access and return a
   session token.
6. WHEN the cosine similarity is at or below the threshold, THE CLIP_Authenticator SHALL deny access
   and return an authentication failure response without revealing the stored embedding.
7. THE CLIP_Authenticator SHALL use a frozen pre-trained CLIP model; CLIP weights SHALL NOT be
   fine-tuned.

---

### Requirement 12: Secure User Credential Storage

**User Story:** As a security engineer, I want user embeddings to be stored securely, so that
credential data cannot be trivially reversed or leaked.

#### Acceptance Criteria

1. THE User_DB SHALL store user embeddings as normalized float32 vectors alongside a username
   identifier.
2. THE User_DB SHALL NOT store raw images, raw passwords, or any reversible representation of
   authentication inputs.
3. THE User_DB SHALL persist user records to disk in a format that survives server restarts.
4. WHEN a username that already exists is registered, THE User_DB SHALL return a conflict error
   rather than silently overwriting the existing record.
5. IF the User_DB file is missing or corrupted on startup, THEN THE User_DB SHALL initialize an
   empty store and log a warning rather than crashing.

---

### Requirement 13: Model Integrity Verification

**User Story:** As a security engineer, I want the system to verify the integrity of model weight
files before loading them, so that tampered or corrupted models are rejected.

#### Acceptance Criteria

1. THE System SHALL compute and store the SHA-256 hash of each model weight file at the time of
   training completion.
2. WHEN a model weight file is loaded at inference time, THE System SHALL recompute the SHA-256
   hash of the file and compare it to the stored hash.
3. IF the computed hash does not match the stored hash, THEN THE System SHALL abort loading and
   raise a `ModelIntegrityError` with a descriptive message.
4. THE System SHALL verify integrity for all model components: HAMMER weights, Image_Watermark_Encoder
   weights, Image_Watermark_Decoder weights, Text_Watermark_Encoder weights, and
   Text_Watermark_Decoder weights.

---

### Requirement 14: Next.js API Routes Backend

**User Story:** As a developer, I want REST API endpoints built into the unified Next.js application
that expose authentication and detection functionality, so that the frontend and the backend share a
single deployment with no separate server process.

#### Acceptance Criteria

1. THE API_Server SHALL expose a `POST /api/login` route that accepts a multipart form with a
   profile image file and a password string, and returns a session token on successful authentication.
2. THE API_Server SHALL expose a `POST /api/predict` route that accepts a multipart form with an
   image file and a text string, and returns the detection result including Fake/Real label,
   bounding box, fake token positions, watermark validity, and Trust_Score.
3. WHEN a request to `POST /api/predict` is received without a valid session token, THE API_Server
   SHALL return HTTP 401 Unauthorized.
4. WHEN `POST /api/login` receives credentials that fail CLIP_Authenticator verification, THE
   API_Server SHALL return HTTP 401 Unauthorized without revealing internal error details.
5. WHEN `POST /api/predict` receives a malformed or unsupported image file, THE API_Server SHALL
   return HTTP 422 Unprocessable Entity with a descriptive error message.
6. THE API_Server SHALL respond to `POST /api/predict` within 10 seconds for a single image-text
   pair on the target T4 GPU hardware.
7. THE API_Server SHALL be implemented as Next.js Route Handlers (App Router `route.ts` files) or
   Next.js API Routes (Pages Router `pages/api/` files) within the same Next.js project as the
   Frontend, with no separate FastAPI or Express server process.

---

### Requirement 15: Unified Next.js Frontend and Backend

**User Story:** As an end user, I want a web interface to log in and submit image-text pairs for
deepfake detection, so that I can use the system without writing code, served from a single Next.js
application that also hosts the API routes.

#### Acceptance Criteria

1. THE Frontend SHALL provide a login page with fields for uploading a profile image and entering a
   text password, and a submit button that calls `POST /api/login`.
2. THE Frontend SHALL provide a detection page with fields for uploading an image and entering a
   text caption, and a submit button that calls `POST /api/predict`.
3. WHEN a detection result is received, THE Frontend SHALL display the Fake/Real label, the
   Trust_Score, and the watermark validity status.
4. WHEN a detection result is received and the sample is classified as fake, THE Frontend SHALL
   overlay bounding box highlights on the uploaded image to indicate manipulated regions.
5. WHEN a detection result is received and the sample is classified as fake, THE Frontend SHALL
   highlight the fake token positions within the displayed text caption.
6. WHEN the session token is absent or expired, THE Frontend SHALL redirect the user to the login
   page rather than displaying an error state.
7. THE Frontend SHALL display a loading indicator while awaiting a response from the API_Server.
8. THE Frontend SHALL be implemented as a single Next.js application that co-locates both the UI
   pages and the API routes (`/api/login`, `/api/predict`), with no separate backend server process.

---

### Requirement 16: Watermark Round-Trip Integrity

**User Story:** As a developer, I want to verify that watermarks survive the full encode-decode
cycle, so that the system's consistency checking is reliable.

#### Acceptance Criteria

1. FOR ALL valid image tensors `I` and watermark vectors `m_T`, applying Image_Watermark_Encoder
   followed by Image_Watermark_Decoder SHALL produce `m_T_hat` with NC greater than 0.95 relative
   to `m_T` (round-trip property).
2. FOR ALL valid token embedding tensors `E` and watermark vectors `m_I`, applying
   Text_Watermark_Encoder followed by Text_Watermark_Decoder SHALL produce `m_I_hat` with NC
   greater than 0.95 relative to `m_I` (round-trip property).
3. THE Metrics_Module `compute_psnr` function SHALL satisfy the identity property: when called with
   two identical tensors, THE Metrics_Module SHALL return `+inf` (or a value exceeding 100 dB).
4. THE Metrics_Module `compute_nc` function SHALL satisfy the identity property: when called with
   two identical tensors, THE Metrics_Module SHALL return 1.0.

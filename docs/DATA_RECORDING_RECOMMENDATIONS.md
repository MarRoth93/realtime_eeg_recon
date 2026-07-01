# EEG Data Recording Recommendations for Image Reconstruction

This document provides guidelines for data acquisition and pipeline configuration for single-subject EEG-to-image reconstruction, based on current project standards and 2024-2026 SOTA research.

## 1. Data Quantity Recommendations

| Strategy | Total Trials | Recording Time | Goal |
| :--- | :--- | :--- | :--- |
| **From Scratch** | **16,000 – 20,000** | ~10–12 hours | Training a subject-specific encoder (NICE/CNN) from zero. |
| **Fine-Tuning** | **1,000 – 2,000** | ~1–2 hours | Adapting a pre-trained model to a new subject's SNR and channel profile. |
| **Zero-Shot** | **0 (Zero)** | N/A | Using a "Foundation Model" with immediate inference. |

## 2. Experimental Design (Recording Protocol)

To achieve results comparable to State-of-the-Art (SOTA) research:

*   **Stimulus Duration:** 100ms (fast presentation prevents eye saccades).
*   **Inter-Stimulus Interval (ISI):** 800ms – 1100ms (allows visual evoked potentials to return to baseline).
*   **Training Set:** Use a wide diversity of concepts (at least 1,000 unique objects) with 1–2 images each.
*   **Test Set (Averaging):** Record **20–40 repetitions** of a smaller set (e.g., 50–200 images). 
    *   *Rationale:* EEG is noisy. Averaging trials of the same image drastically increases the Signal-to-Noise Ratio (SNR), which is critical for high-quality reconstruction.

## 3. Pipeline Strategy

*   **Subject-Specific Adapter:** Instead of training the entire encoder (e.g., NICE with 1.1M parameters) on one person, train a small **linear adapter layer** that maps specific channel configurations (e.g., Starstim 32) to the pre-trained latent space.
*   **Two-Stage Reconstruction:**
    *   **Path A (Real-Time):** Focus on **EEG → CLIP Embedding** (<30ms). This enables semantic retrieval and instant feedback.
    *   **Path B (High-Fidelity):** Use the predicted CLIP embedding to drive a **Diffusion Model** (e.g., SDXL-Turbo) asynchronously for photorealistic pixels.
*   **Normalization:** Use **Online Z-Score Normalization** to handle impedance drift during long recording sessions.

## 4. Hardware Adaptation (Starstim 32)

If using 32-channel hardware with a 63/64-channel model (like NICE):
1.  Use **Spatial Interpolation** to map 32 channels to the expected 63-channel positions.
2.  Alternatively, implement the **ENIGMA Subject-Specific Adapter** approach to map 32-channel raw data into a "Universal Brain Space" before encoding.

## 5. Summary Plan for Single Subject

For a practical yet high-quality result, I recommend:
1.  **Session Duration:** 2 hours.
2.  **Training:** 1,800 trials (unique images).
3.  **Testing:** 20 images repeated 10 times each (200 trials).
4.  **Method:** Fine-tune a pre-trained **NICE/ATM encoder** rather than training from scratch.

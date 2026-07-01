from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from maryam_rt.hierarchical.atms_pipeline.atms_utils import ATMS
from maryam_rt.hierarchical.atms_pipeline.custom_pipeline_low_level import Generator4Embeds
from maryam_rt.hierarchical.atms_pipeline.diffusion_prior import (
    DiffusionPriorUNet,
    DiffusionPriorUNet_FDN,
    Pipe,
)


class HighLevelRefiner:
    """Slow path semantic refinement that runs off the realtime loop."""

    def __init__(
        self,
        atms_checkpoint: str | Path,
        prior_checkpoint: str | Path,
        subject_id: int = 0,
        use_fdn: bool = True,
        num_prior_steps: int = 50,
        num_sdxl_steps: int = 10,
        guidance_scale: float = 5.0,
        img2img_strength: float = 0.8,
        text_prompt: str = "a photo of an object",
        device: str = "cuda",
    ) -> None:
        self.device = device
        self.subject_id = subject_id
        self.num_prior_steps = num_prior_steps
        self.guidance_scale = guidance_scale
        self.text_prompt = text_prompt

        self.atms = ATMS(num_subjects=1).to(device)
        atms_state = torch.load(atms_checkpoint, map_location=device, weights_only=True)
        exclude = {"subject_wise_linear.1.weight", "subject_wise_linear.1.bias"}
        atms_state = {key: value for key, value in atms_state.items() if key not in exclude}
        self.atms.load_state_dict(atms_state, strict=False)
        self.atms.eval()

        if use_fdn:
            self.prior = DiffusionPriorUNet_FDN(cond_dim=1024, dropout=0.1).to(device)
        else:
            self.prior = DiffusionPriorUNet(cond_dim=1024, dropout=0.1).to(device)
        self.prior.load_state_dict(torch.load(prior_checkpoint, map_location=device, weights_only=True))
        self.prior.eval()
        self.prior_pipe = Pipe(self.prior, device=device)

        self.generator = Generator4Embeds(
            num_inference_steps=num_sdxl_steps,
            device=device,
            img2img_strength=img2img_strength,
        )

    @torch.inference_mode()
    def refine(
        self,
        x250: torch.Tensor | np.ndarray,
        low_level_image: Image.Image | np.ndarray,
        text_prompt: str | None = None,
    ) -> Image.Image:
        if not isinstance(x250, torch.Tensor):
            x250 = torch.as_tensor(x250, dtype=torch.float32)
        if x250.ndim == 2:
            x250 = x250.unsqueeze(0)
        x250 = x250.to(self.device, dtype=torch.float32)

        subject_ids = torch.full(
            (x250.shape[0],),
            self.subject_id,
            dtype=torch.long,
            device=self.device,
        )
        eeg_emb = self.atms(x250, subject_ids)
        if eeg_emb.ndim == 3 and eeg_emb.shape[1] == 1:
            eeg_emb = eeg_emb[:, 0, :]

        prior_output = self.prior_pipe.generate(
            c_embeds=eeg_emb,
            num_inference_steps=self.num_prior_steps,
            guidance_scale=self.guidance_scale,
        )

        if isinstance(low_level_image, np.ndarray):
            low_level_image = Image.fromarray(low_level_image.astype(np.uint8))
        low_level_image = low_level_image.resize((512, 512))

        prompt = self.text_prompt if text_prompt is None else text_prompt

        return self.generator.generate(
            prior_output,
            text_prompt=prompt,
            low_level_image=low_level_image,
        )

from .base import BasePolicy
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError

from gr00t.data.dataset import ModalityConfig
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.schema import DatasetMetadata
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.model.gr00t_n1 import GR00T_N1_5

class GR00TPolicy(BasePolicy):
    def __init__(
        self, 
        task_name,
        model_path: str,
        embodiment_tag: Union[str, EmbodimentTag],
        modality_config: Dict[str, ModalityConfig],
        modality_transform: ComposedModalityTransform,
        denoising_steps: Optional[int] = None,
        device: Union[int, str] = "cuda" if torch.cuda.is_available() else "cpu",
        load_meta: bool = True,
    ) -> None:
        super().__init__(task_name)
        try:
            # NOTE(YL) this returns the local path to the model which is normally
            # saved in ~/.cache/huggingface/hub/
            model_path = snapshot_download(model_path, repo_type="model")
            # HFValidationError, RepositoryNotFoundError
        except (HFValidationError, RepositoryNotFoundError):
            print(
                f"Model not found or avail in the huggingface hub. Loading from local path: {model_path}"
            )

        self._modality_config = modality_config
        self._modality_transform = modality_transform
        self._modality_transform.eval()  # set this to eval mode
        self.model_path = Path(model_path)
        self.device = device

        # Convert string embodiment tag to EmbodimentTag enum if needed
        if isinstance(embodiment_tag, str):
            self.embodiment_tag = EmbodimentTag(embodiment_tag)
        else:
            self.embodiment_tag = embodiment_tag

        # Load model
        self._load_model(model_path)
        # Load transforms
        if load_meta:
            self._load_metadata(self.model_path / "experiment_cfg")
        # Load horizons
        self._load_horizons()

        if denoising_steps is not None:
            if hasattr(self.model, "action_head") and hasattr(
                self.model.action_head, "num_inference_timesteps"
            ):
                self.model.action_head.num_inference_timesteps = denoising_steps
                print(f"Set action denoising steps to {denoising_steps}")
    
    def act(self, observations, **kwargs):

        return
    
    def reset(self):
        target_position = [
            0.34906611,
            0.34987221,
            0,
            0.436332313,
            -0.66857928,
            0.67156327,
            0.2008844,
            -0.20287371,
            0.27921745,
            -0.282218840,
            -1.28203404,
            1.28208637,
            0.84163094,
            -0.84068865,
            1.51518357,
            -1.51710308,
            -0.18715125,
            0.18636601,
            1,
            -1,
            1,
            -1,
            0,
            1,
            0,
            1,
            0,
            0,
            1,
            1,
            1,
            1,
            0,
            0,
        ]
        return target_position

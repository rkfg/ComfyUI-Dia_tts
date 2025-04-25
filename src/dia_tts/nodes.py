import logging
import random
import numpy as np
import torch
from .dia import Dia
from .dia.model import ComputeDtype
import comfy.model_management as mm
import torchaudio as ta

DEFAULT_SAMPLE_RATE = 44100


def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN (if used)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DiaModelLoader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (
                    "STRING",
                    {"tooltip": "HF model name", "default": "nari-labs/Dia-1.6B"},
                ),
            },
        }

    RETURN_TYPES = ("TTSMODEL",)
    # RETURN_NAMES = ("image_output_name",)
    FUNCTION = "load"

    # OUTPUT_NODE = False
    # OUTPUT_TOOLTIPS = ("",) # Tooltips for the output node

    CATEGORY = "Dia TTS"

    def load(self, model_name):
        logging.info(f"Loading TTS model from {model_name}")
        model = Dia.from_pretrained(model_name, compute_dtype=ComputeDtype.BFLOAT16)
        return (model,)


class DiaSampler:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": (
                    "STRING",
                    {"default": "[S1] Hello!\n[S2] Hi!", "multiline": True},
                ),
                "max_tokens": ("INT", {"default": 3072, "min": 860, "max": 16384}),
                "temperature": (
                    "FLOAT",
                    {"default": 1.3, "min": 0.7, "max": 1.5, "step": 0.05},
                ),
                "cfg_scale": ("FLOAT", {"default": 3, "min": 1, "max": 5, "step": 0.1}),
                "top_p": (
                    "FLOAT",
                    {"default": 0.95, "min": 0.8, "max": 1, "step": 0.01},
                ),
                "top_pcfg_filter_top_k": (
                    "INT",
                    {"default": 30, "min": 15, "max": 50, "step": 1},
                ),
                "speed_factor": (
                    "FLOAT",
                    {"default": 0.94, "min": 0.8, "max": 1, "step": 0.02},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "use_torch_compile": ("BOOLEAN", {"default": True}),
                "model": ("TTSMODEL",),
            },
            "optional": {"audio_prompt": ("AUDIO",)},
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "sample"
    CATEGORY = "Dia TTS"

    def sample(
        self,
        text,
        max_tokens,
        temperature,
        cfg_scale,
        top_p,
        top_pcfg_filter_top_k,
        speed_factor: float,
        seed: int,
        use_torch_compile: bool,
        model: Dia,
        audio_prompt: dict = None,
    ):
        mm.unload_all_models()
        mm.cleanup_models()
        mm.soft_empty_cache()
        device = mm.get_torch_device()
        set_seed(seed)
        if audio_prompt is not None:
            sr, audio = audio_prompt["sample_rate"], audio_prompt["waveform"]
            if sr != DEFAULT_SAMPLE_RATE:
                audio = ta.functional.resample(audio, sr, DEFAULT_SAMPLE_RATE)

            audio = model.dac_model.preprocess(audio.to(device), DEFAULT_SAMPLE_RATE)
            _, encoded_frame, _, _, _ = model.dac_model.encode(audio)  # 1, C, T
            audio_prompt = encoded_frame.squeeze(0).transpose(0, 1)

        with torch.inference_mode():
            output_audio = model.generate(
                text=text,
                max_tokens=max_tokens,
                cfg_scale=cfg_scale,
                temperature=temperature,
                top_p=top_p,
                use_torch_compile=use_torch_compile,
                audio_prompt=audio_prompt,
            )
            original_len = len(output_audio)
            # Ensure speed_factor is positive and not excessively small/large to avoid issues
            speed_factor = max(0.1, min(speed_factor, 5.0))
            target_len = int(
                original_len / speed_factor
            )  # Target length based on speed_factor
            if (
                target_len != original_len and target_len > 0
            ):  # Only interpolate if length changes and is valid
                x_original = np.arange(original_len)
                x_resampled = np.linspace(0, original_len - 1, target_len)
                output_audio = torch.tensor(
                    np.interp(x_resampled, x_original, output_audio)
                )

        output_audio = output_audio.reshape((1, 1, -1))
        mm.soft_empty_cache()
        return ({"waveform": output_audio, "sample_rate": DEFAULT_SAMPLE_RATE},)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {"DiaModelLoader": DiaModelLoader, "DiaSampler": DiaSampler}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "DiaModelLoader": "Load TTS model",
    "DiaSampler": "Generate speech",
}

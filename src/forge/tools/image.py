"""Image generation tools for Forge agents.

Two implementations:

* **LocalImageTool** — offline Stable Diffusion via HuggingFace ``diffusers``.
  Install: ``pip install diffusers accelerate``
  Auto-selects device: Apple Silicon MPS → CUDA → CPU.
  Default model: ``stabilityai/sdxl-turbo`` (4 steps, ~7 GB, fast).

* **DalleImageTool** — OpenAI DALL-E 3 via API.
  Requires: ``openai`` (already in requirements.txt) + ``OPENAI_API_KEY``.

Both tools save the generated image to disk and return the file path,
so the agent can report it to the user.
"""

from __future__ import annotations

import urllib.request
import uuid
from pathlib import Path
from typing import Any

from forge.permission.manager import PermissionManager
from forge.tools.base import ForgeTool

# Default output directory (current working directory)
_DEFAULT_OUTPUT_DIR = "."


def _extract_prompt(tool_input: Any) -> str:
    """Pull the prompt string out of whatever the LLM passed."""
    if isinstance(tool_input, dict):
        return str(tool_input.get("tool_input") or tool_input.get("prompt") or "")
    return str(tool_input)


def _save_path(output_dir: str, ext: str = "png") -> str:
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)
    return str(p / f"{uuid.uuid4().hex[:8]}.{ext}")


# ---------------------------------------------------------------------------
# LocalImageTool — Stable Diffusion via diffusers (offline after first download)
# ---------------------------------------------------------------------------

class LocalImageTool(ForgeTool):
    """Generate images locally with Stable Diffusion (diffusers).

    Requires: ``pip install diffusers accelerate``

    Args:
        model_id:    HuggingFace model ID.
                     Default: ``stabilityai/sdxl-turbo`` (fast, 4 steps).
        steps:       Number of inference steps (default: 4 for SDXL-Turbo).
        output_dir:  Directory to save generated images.
        permissions: Forge permission manager.
    """

    name: str = "image"
    description: str = (
        "Generate an image from a text description. "
        "Pass a detailed prompt describing the scene, style, lighting, etc. "
        "Returns the file path of the saved image."
    )
    input_description: str = (
        "The image prompt as a plain text string — NOT a schema type object. "
        "Example: 'a wolf howling at the full moon, oil painting, "
        "dramatic lighting, dark forest background, cinematic'. "
        "Describe subject, art style, lighting, mood and colors."
    )

    def __init__(
        self,
        permissions: PermissionManager | None = None,
        model_id: str = "stabilityai/sdxl-turbo",
        steps: int = 4,
        output_dir: str = _DEFAULT_OUTPUT_DIR,
    ) -> None:
        super().__init__(permissions)
        self._model_id = model_id
        self._steps = steps
        self._output_dir = output_dir
        self._pipeline: Any = None

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        try:
            import torch  # noqa: PLC0415
            from diffusers import AutoPipelineForText2Image  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "diffusers not installed. Run: pip install diffusers accelerate"
            ) from exc

        if torch.backends.mps.is_available():
            device, dtype = "mps", torch.float16
        elif torch.cuda.is_available():
            device, dtype = "cuda", torch.float16
        else:
            device, dtype = "cpu", torch.float32

        print(
            f"[image] loading {self._model_id} on {device} "
            "(cached after first download)…",
        )
        self._pipeline = AutoPipelineForText2Image.from_pretrained(
            self._model_id,
            torch_dtype=dtype,
        ).to(device)
        print("[image] model ready")

    def run(self, tool_input: Any) -> Any:
        prompt = _extract_prompt(tool_input)
        if not prompt:
            return "Error: prompt is required"

        try:
            self._load()
        except RuntimeError as exc:
            return str(exc)
        except Exception as exc:
            return f"Error loading model: {exc}"

        try:
            result = self._pipeline(
                prompt=prompt,
                num_inference_steps=self._steps,
                guidance_scale=0.0,  # SDXL-Turbo requires guidance_scale=0
            )
            image = result.images[0]
        except Exception as exc:
            return f"Error generating image: {exc}"

        path = _save_path(self._output_dir)
        image.save(path)
        return f"Image saved: {path}"


# ---------------------------------------------------------------------------
# DalleImageTool — OpenAI DALL-E via API
# ---------------------------------------------------------------------------

class DalleImageTool(ForgeTool):
    """Generate images using the OpenAI DALL-E API.

    Requires: ``openai`` package + ``OPENAI_API_KEY`` environment variable.

    Args:
        api_key:     Override for the OpenAI API key (default: env var).
        model:       DALL-E model (``dall-e-3`` or ``dall-e-2``).
        size:        Image size string (e.g. ``"1024x1024"``).
        quality:     ``"standard"`` or ``"hd"`` (DALL-E 3 only).
        output_dir:  Directory to save downloaded images.
        permissions: Forge permission manager.
    """

    name: str = "image"
    description: str = (
        "Generate a high-quality image from a text description using DALL-E. "
        "Pass a detailed prompt describing the scene, style, and composition. "
        "Returns the file path of the saved image."
    )
    input_description: str = (
        "The image prompt as a plain text string — NOT a schema type object. "
        "Example: 'a wolf howling at the full moon, oil painting, "
        "dramatic lighting, dark forest background, cinematic'. "
        "Describe subject, art style, lighting, mood and colors."
    )

    def __init__(
        self,
        permissions: PermissionManager | None = None,
        api_key: str | None = None,
        model: str = "dall-e-3",
        size: str = "1024x1024",
        quality: str = "standard",
        output_dir: str = _DEFAULT_OUTPUT_DIR,
    ) -> None:
        super().__init__(permissions)
        self._api_key = api_key
        self._model = model
        self._size = size
        self._quality = quality
        self._output_dir = output_dir

    def run(self, tool_input: Any) -> Any:
        prompt = _extract_prompt(tool_input)
        if not prompt:
            return "Error: prompt is required"

        try:
            from openai import OpenAI  # noqa: PLC0415
        except ImportError:
            return "Error: openai not installed. Run: pip install openai"

        kwargs: dict[str, Any] = {}
        if self._api_key:
            kwargs["api_key"] = self._api_key

        try:
            client = OpenAI(**kwargs)
            response = client.images.generate(
                model=self._model,
                prompt=prompt,
                size=self._size,  # type: ignore[arg-type]
                quality=self._quality,  # type: ignore[arg-type]
                n=1,
            )
        except Exception as exc:
            return f"Error calling DALL-E API: {exc}"

        image_url = response.data[0].url
        if not image_url:
            return "Error: no image URL in response"

        path = _save_path(self._output_dir)
        try:
            urllib.request.urlretrieve(image_url, path)
        except Exception as exc:
            return f"Error downloading image: {exc}"

        return f"Image saved: {path}"

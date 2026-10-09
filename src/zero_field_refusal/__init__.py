"""Zero-field fine-tuning: how much does the optimizer alone erode refusals?"""

from .paths import load_env

# First, before any module imports huggingface_hub, which reads HF_HOME and HF_TOKEN on import.
load_env()

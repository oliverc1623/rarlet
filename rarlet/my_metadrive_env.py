from typing import Any

from metadrive.envs import MetaDriveEnv


# --- 1. Define the Custom Environment Class ---
class CustomMetaDriveEnv(MetaDriveEnv):
    """Custom MetaDrive Environment."""

    def __init__(self, config: dict):
        # Initialize the parent MetaDriveEnv
        super().__init__(config)

    # --- 2. Override the reset method ---
    def reset(self, seed: int, options: Any = None) -> tuple:  # noqa: ARG002
        """Call the parent reset method."""
        obs, info = super().reset(seed)
        return obs, info

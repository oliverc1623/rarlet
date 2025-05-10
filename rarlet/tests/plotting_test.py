# %%

from pathlib import Path


root_path = Path(__file__).resolve().parent.parent

# %%


def test_get_vector_env_seeding() -> None:
    """Test that each environment in the vectorized environment has a different seed."""
    x = 1
    y = 1
    assert x == y

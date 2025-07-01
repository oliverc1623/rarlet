# %% Sac Protagonist Eval

import gc  # garbage collect library
import math
import os
import pathlib
from pathlib import Path

import gymnasium as gym
import numpy as np
import scenic
import torch
import torch.nn.functional as f
from gymnasium import spaces
from IPython.display import Image
from scenic.gym.envs.scenic_gym import ScenicGymEnv
from scenic.simulators.metadrive.simulator import MetaDriveSimulator
from torch import nn

import wandb


gc.collect()
with torch.no_grad():
    torch.cuda.empty_cache()


os.environ["MUJOCO_GL"] = "egl"  # must precede any mujoco/gym import

# %%
LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    """A neural network that approximates the policy for reinforcement learning."""

    def __init__(self, env: gym.Env, n_obs: int, n_act: int, device: str | None = None):
        super().__init__()
        self.fc1 = nn.Linear(n_obs, 256, device=device)
        self.fc2 = nn.Linear(256, 256, device=device)
        self.fc_mean = nn.Linear(256, n_act, device=device)
        self.fc_logstd = nn.Linear(256, n_act, device=device)
        # action rescaling
        action_space = env.action_space
        self.register_buffer(
            "action_scale",
            torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32, device=device),
        )

    def forward(self, x: torch.tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        x = x.view(x.shape[0], -1)  # flatten the input
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x: torch.tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action from the policy network."""
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


# %%
seed = 318
scenario_file = "scenarios/drive.scenic"
model = "scenic.simulators.metadrive.model"
carla_map = "maps/Town06.net.xml"
run_id = "ahxb61qd"

scenario = scenic.scenarioFromFile(
    scenario_file,
    model=model,
    mode2D=True,
)

env = ScenicGymEnv(
    scenario,
    MetaDriveSimulator(
        timestep=0.02,
        sumo_map=pathlib.Path(carla_map),
        render=False,
        real_time=False,
        render3D=False,
    ),
    observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(258,)),
    action_space=spaces.Box(low=-1, high=1, shape=(2,)),
    max_steps=600,
)

video_dest_dir = Path("videos/inference/test/")
video_dest_dir.mkdir(exist_ok=True)
video_dest = f"{video_dest_dir}/{run_id}.gif"

n_act = math.prod(env.action_space.shape)
n_obs = math.prod(env.observation_space.shape)
device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
wandb.login(key="82555a3ad6bd991b8c4019a5a7a86f61388f6df1")
api = wandb.Api()

# %%

pretrained_run = api.run(f"rarlet/{run_id}")
actor_file = next(f.name for f in pretrained_run.files() if f.name.endswith("_actor.pt"))
model_weights = wandb.restore(actor_file, run_path=f"rarlet/{run_id}", root="models")

# %%
with torch.no_grad():
    protagonist = Actor(env, device=device, n_act=n_act, n_obs=n_obs)
    protagonist.load_state_dict(
        torch.load(model_weights.name),
    )
    protagonist.eval()

# %%
try:
    total_reward = 0
    obs, _ = env.reset(seed=seed)
    done = False
    trunc = False
    while not done and not trunc:
        obs = torch.as_tensor(obs, device=device, dtype=torch.float).unsqueeze(0)
        action, _, _ = protagonist.get_action(obs)
        action = action.squeeze().detach().cpu().numpy()

        obs, reward, done, trunc, info = env.step([0, 1.0])
        total_reward += reward
    assert env
    env.simulator.client.top_down_renderer.generate_gif(video_dest)
    print(info)
    print("episode_reward", total_reward)
    total_reward = 0
finally:
    env.close()

# %%
Image(Path.open(video_dest, "rb").read())

# %% Sac Adversary Eval

import math
from pathlib import Path

import gymnasium as gym
import torch
import torch.nn.functional as f
import wandb
from IPython.display import Image
from metadrive.policy.idm_policy import IDMPolicy
from my_metadrive_env import AdversaryMetaDriveEnv
from torch import nn


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
seed = 1
num_envs = 1

env = AdversaryMetaDriveEnv(
    dict(
        map="SS",
        horizon=125,
        # scenario setting
        random_spawn_lane_index=True,
        num_scenarios=8,
        start_seed=1,
        traffic_density=0.5,
        accident_prob=0.0,
        log_level=50,
        vehicle_config=dict(
            spawn_longitude=70,
            spawn_velocity=(10, 0),
        ),
        traffic_mode="basic",
        forward_reward=1.0,
        expert_vehicle_ratio=0.0,
    ),
)

n_act = math.prod(env.action_space.shape)
n_obs = math.prod(env.observation_space.shape)
device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
wandb.login(key="82555a3ad6bd991b8c4019a5a7a86f61388f6df1")

# %%
best_model = wandb.restore("SS__advo-train__1__True__True_actor.pt", run_path="rarlet/v7qpltkx")

# %%
protagonist = Actor(env, device=device, n_act=n_act, n_obs=n_obs)
protagonist.load_state_dict(
    torch.load(best_model.name),
)

# %%
seed = 2
try:
    total_reward = 0
    obs, _ = env.reset(seed)
    done = False
    trunc = False
    while not done and not trunc:
        obs = torch.as_tensor(obs, device=device, dtype=torch.float).unsqueeze(0)
        action, _, _ = protagonist.get_action(obs)
        action = action.squeeze().detach().cpu().numpy()

        obs, reward, done, trunc, info = env.step(action)
        total_reward += reward
        ret = env.render(
            mode="topdown",
            screen_record=True,
            window=False,
            screen_size=(400, 400),
            camera_position=(100, 7),
            scaling=2,
            text={
                "Timestep": env.episode_step,
                "Reward": reward,
                "Victim crashes": info["behind_crashes"],
                "max_steps": info["max_step"],
            },
        )
    print(info)
    print("episode_reward", total_reward)
    total_reward = 0
    env.top_down_renderer.generate_gif(f"movies/sac_adversary_scene_{seed:02d}.gif")
finally:
    env.close()

# %%
Image(Path.open(f"movies/sac_adversary_scene_{seed:02d}.gif", "rb").read())

# %% RUN IDM BASELINE
device = "cuda" if torch.cuda.is_available() else "cpu"
seed = 2
num_envs = 1

env = AdversaryMetaDriveEnv(
    dict(
        map="SS",
        horizon=125,
        # scenario setting
        random_spawn_lane_index=True,
        agent_policy=IDMPolicy,
        num_scenarios=8,
        start_seed=1,
        traffic_density=0.5,
        accident_prob=0.0,
        log_level=50,
        vehicle_config=dict(
            spawn_longitude=70,
            spawn_velocity=(10, 0),
        ),
        traffic_mode="basic",
        forward_reward=1.0,
        expert_vehicle_ratio=0.0,
    ),
)

# %%
seed = 2
try:
    total_reward = 0
    obs, _ = env.reset(seed)
    done = False
    trunc = False
    while not done and not trunc:
        obs, reward, done, trunc, info = env.step([0, 0])
        total_reward += reward
        ret = env.render(
            mode="topdown",
            screen_record=True,
            window=False,
            screen_size=(400, 400),
            camera_position=(100, 7),
            scaling=2,
            text={
                "Timestep": env.episode_step,
                "Reward": reward,
                "Victim crashes": info["behind_crashes"],
                "max_steps": info["max_step"],
            },
        )
    print(info)
    print("episode_reward", total_reward)
    total_reward = 0
    env.top_down_renderer.generate_gif(f"movies/IDM_adversary_baseline_scene_{seed:02d}.gif")
finally:
    env.close()

# %%
Image(Path.open(f"movies/IDM_adversary_baseline_scene_{seed:02d}.gif", "rb").read())


# %%

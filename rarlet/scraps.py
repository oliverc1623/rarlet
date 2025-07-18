# ruff: noqa

# %%
import cProfile
import pathlib
import pstats
from io import StringIO

import metadrive.component
import metadrive.component.vehicle
import numpy as np
import scenic
from gymnasium import spaces
from scenic.gym import ScenicGymEnv
from scenic.simulators.metadrive import MetaDriveSimulator
from stable_baselines3.common.utils import set_random_seed
import matplotlib.pyplot as plt

# %%

scenario = scenic.scenarioFromFile(
    "scenarios/drive.scenic",
    model="scenic.simulators.metadrive.model",
    mode2D=True,
)

env = ScenicGymEnv(
    scenario,
    MetaDriveSimulator(timestep=0.1, sumo_map=pathlib.Path("maps/Town06.net.xml"), render=True, real_time=False),
    observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(258,)),
    action_space=spaces.Box(low=-1, high=1, shape=(2,)),
    max_steps=600,
)

# %%

total_reward = 0
obs, _ = env.reset(seed=1)
done = False
trunc = False
for i in range(600):
    obs, reward, done, trunc, info = env.step([0.5, 1.0])
    total_reward += reward
    print(f"step {i} reward {reward}, done: {done}")
    if done or trunc:
        break
print(f"done: {done}, trunc: {trunc}")
print("terminate reward", reward)
print("total reward", total_reward)
assert env
print(info)

# %%

env.close()


# %%
# Assuming 'env' is your Gymnasium environment instance
profiler = cProfile.Profile()
profiler.enable()

env.reset()  # The function call you want to profile

profiler.disable()
s = StringIO()
sortby = "cumulative"  # or 'tottime'
ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())

# %%
env.reset()
a = env.action_space.sample()

# Assuming 'env' is your Gymnasium environment instance
profiler = cProfile.Profile()
profiler.enable()

env.step(a)  # The function call you want to profile

profiler.disable()
s = StringIO()
sortby = "cumulative"  # or 'tottime'
ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())


# %%
def main() -> None:
    """Run RL training."""
    set_random_seed(0)

    scenario = scenic.scenarioFromFile(
        "scenarios/protagonsit.scenic",
        model="scenic.simulators.metadrive.model",
        mode2D=True,
    )

    env = ScenicGymEnv(
        scenario,
        MetaDriveSimulator(timestep=0.02, sumo_map=pathlib.Path("maps/Town06.net.xml"), render=False, real_time=False),
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(258,)),
        action_space=spaces.Box(low=-1, high=1, shape=(2,)),
        max_steps=600,
    )
    env.reset()

    while True:
        action = env.action_space.sample()
        obs, reward, done, trunc, info = env.step(action)
        print(f"reward: {reward}")
        if done:
            break

    env.close()


if __name__ == "__main__":
    main()


# %%
from metadrive.envs import BaseEnv
from metadrive.manager.sumo_map_manager import SumoMapManager
from metadrive.obs.state_obs import LidarStateObservation


# %%
class DriveEnv(BaseEnv):
    def reward_function(self, agent):
        """Dummy reward function."""
        return 0, {}

    def cost_function(self, agent):
        """Dummy cost function."""
        return 0, {}

    def done_function(self, agent):
        """Dummy done function."""
        return False, {}

    def get_single_observation(self):
        """Dummy observation function."""
        return LidarStateObservation(self.config)

    def setup_engine(self):
        """Setup the engine for MetaDrive."""
        super().setup_engine()
        self.engine.register_manager("map_manager", SumoMapManager("maps/Town06.net.xml"))


# %%
vehicle_config = {}
vehicle_config["spawn_position_heading"] = [
    (0.0, 0.0),
    0.0,
]
vehicle_config["lane_line_detector"] = dict(
    num_lasers=10,
    distance=20,
)
client = DriveEnv(
    dict(
        vehicle_config=vehicle_config,
        use_render=False,
        use_mesh_terrain=False,
    )
)
client.config["sumo_map"] = "maps/Town06.net.xml"

# %%
client.reset()

# Assuming 'env' is your Gymnasium environment instance
profiler = cProfile.Profile()
profiler.enable()

client.reset()

profiler.disable()
s = StringIO()
sortby = "cumulative"  # or 'tottime'
ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())

# %%
from metadrive.envs import MetaDriveEnv
from metadrive.policy.lange_change_policy import LaneChangePolicy
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import Monitor
from metadrive.component.map.base_map import BaseMap
from metadrive.utils.doc_utils import generate_gif
from IPython.display import clear_output, Image


def create_env(need_monitor=False):
    env = MetaDriveEnv(
        dict(
            map="S",
            horizon=500,
            # scenario setting
            random_spawn_lane_index=True,
            num_scenarios=1,
            start_seed=1,
            traffic_density=0.1,
            accident_prob=0,
            log_level=50,
        )
    )
    if need_monitor:
        env = Monitor(env)
    env.action_space.seed(0)
    return env


# %%
env = create_env()
env.reset()
ret = env.render(mode="topdown", window=False, screen_size=(600, 600), camera_position=(50, 50))
env.close()
plt.axis("off")
plt.imshow(ret)

# %%

env.reset(seed=1)
ret = env.render(mode="topdown", window=False, screen_size=(600, 600), camera_position=(50, 50))
env.close()
plt.axis("off")
plt.imshow(ret)

# %%

from my_metadrive_env import CustomMetaDriveEnv
from IPython.display import clear_output, Image
from metadrive.component.map.pg_map import MapGenerateMethod
from metadrive.component.map.base_map import BaseMap

# %%
map_config = {
    BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_SEQUENCE,
    BaseMap.GENERATE_CONFIG: "SSS",
    BaseMap.LANE_WIDTH: 3.5,
    BaseMap.LANE_NUM: 2,
}

env = CustomMetaDriveEnv(
    dict(
        map_config=map_config,
        horizon=500,
        # scenario setting
        random_spawn_lane_index=False,
        num_scenarios=8,
        start_seed=31,
        traffic_density=0.3,
        accident_prob=0.0,
        log_level=50,
    )
)
# %%
try:
    env.reset(31)
    for _ in range(125):
        _, r, d, _, info = env.step([0, 1])  # ego car is static
        env.render(
            mode="topdown",
            window=False,
            screen_size=(400, 400),
            camera_position=(100, 7),
            scaling=2,
            screen_record=True,
            text={
                "Has vehicle": bool(len(env.engine.traffic_manager.spawned_objects)),
                "Timestep": env.episode_step,
                "Reward": f"{r:0.2f}",
                "done": d,
            },
        )
    assert env
    env.top_down_renderer.generate_gif()
finally:
    env.close()
    clear_output()
Image(open("demo.gif", "rb").read())


# %%

from my_metadrive_env import AdversaryMetaDriveEnv
from IPython.display import clear_output, Image
import gymnasium as gym

# %%


def make_env(seed: int) -> callable:
    """Create and seed environments."""

    def thunk() -> gym.Env:
        """Create a gym environment."""
        env = AdversaryMetaDriveEnv(
            dict(
                map="SS",
                horizon=125,
                # scenario setting
                random_spawn_lane_index=True,
                num_scenarios=2,
                start_seed=seed,
                traffic_density=0.2,
                vehicle_config=dict(
                    spawn_longitude=70,
                    spawn_velocity=(10, 0),
                ),
                accident_prob=0.0,
                log_level=50,
                init_velo=(10, 0),
                traffic_mode="basic",
                speed_reward=1.0,
            )
        )
        return env

    return thunk


seed = 1
num_envs = 2
env = gym.vector.AsyncVectorEnv(
    [make_env(seed + i) for i in range(num_envs)],
    autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
)

# %%

env.reset()

f1 = env.env_fns[0]()
f1.reset(1)
fr1 = f1.render(mode="topdown", window=False, screen_size=(400, 400), camera_position=(100, 7), scaling=2)
f1.close()


f2 = env.env_fns[1]()
f2.reset(2)
fr2 = f2.render(mode="topdown", window=False, screen_size=(400, 400), camera_position=(100, 7), scaling=2)
f2.close()

# %%

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(20, 10))  # You can adjust the figsize as needed
axes[0].imshow(fr1)
axes[0].axis("off")  # Turn off axis
axes[0].set_title("seed 1")
axes[1].imshow(fr2)
axes[1].axis("off")  # Turn off axis
axes[1].set_title("seed 2")

# %%
try:
    env.reset(0)
    for i in range(210):
        _, r, d, t, info = env.step([1, 0.2])  # ego car is static
        env.render(
            mode="topdown",
            window=False,
            screen_size=(400, 400),
            camera_position=(100, 7),
            scaling=2,
            screen_record=True,
            text={
                "Has vehicle": bool(len(env.engine.traffic_manager.spawned_objects)),
                "Timestep": env.episode_step,
                "Reward": f"{r:0.2f}",
            },
        )
    assert env
    env.top_down_renderer.generate_gif()
finally:
    env.close()
    clear_output()
Image(open("demo.gif", "rb").read())

# %%

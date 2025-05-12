# ruff: noqa

# %%
import cProfile
import pathlib
import pstats
from io import StringIO

import numpy as np
import scenic
from gymnasium import spaces
from scenic.gym import ScenicGymEnv
from scenic.simulators.metadrive import MetaDriveSimulator
from stable_baselines3.common.utils import set_random_seed


# %%


def make_env() -> callable:
    """Create a function that returns a new environment instance."""

    def thunk() -> ScenicGymEnv:
        scenario = scenic.scenarioFromFile(
            "../scenarios/protagonsit.scenic",
            model="scenic.simulators.metadrive.model",
            mode2D=True,
        )

        env = ScenicGymEnv(
            scenario,
            MetaDriveSimulator(timestep=0.02, sumo_map=pathlib.Path("../maps/Town06.net.xml"), render=False, real_time=False),
            observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(258,)),
            action_space=spaces.Box(low=-1, high=1, shape=(2,)),
            max_steps=700,
        )
        return env

    return thunk


# %%

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

# %%
env.reset()  # The function call you want to profile

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

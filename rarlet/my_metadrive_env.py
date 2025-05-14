from typing import Any

import numpy as np
from metadrive.component.vehicle.vehicle_type import DefaultVehicle
from metadrive.envs import MetaDriveEnv
from metadrive.manager import BaseManager
from metadrive.policy.idm_policy import IDMPolicy


class CustomMetaDriveEnv(MetaDriveEnv):
    """Custom MetaDrive Environment."""

    def __init__(self, config: dict):
        super().__init__(config)

    def reset(self, seed: int, options: Any = None) -> tuple:  # noqa: ARG002
        """Call the parent reset method."""
        obs, info = super().reset(seed)
        return obs, info


def clip(a: float, low: float, high: float) -> float:
    """Clip a value to a specified range."""
    return min(max(a, low), high)


class MovingExampleManager(BaseManager):
    """A custom manager for the MetaDrive environment."""

    def __init__(self):
        super().__init__()
        self.generated_v = None
        self.rng = np.random.default_rng()

        # ===== Access the parameters in engine.global_config =====
        self.num_idm_victims = self.engine.global_config["num_idm_victims"]

    """Customize manager to spawn vehicles."""

    def before_step(self) -> None:
        """Set actions for all spawned objects."""
        for obj_id, obj in self.spawned_objects.items():
            p = self.get_policy(obj_id)
            obj.before_step(p.act())  # set action

    def reset(self) -> None:
        """Reset the environment."""
        for i in range(self.num_idm_victims):
            # spawn victim vehicles
            lat_offset = self.rng.uniform(0, 3 * 3)
            obj = self.spawn_object(
                DefaultVehicle,
                vehicle_config=dict(),
                position=(i * 8, lat_offset),
                heading=0,
            )
            self.add_policy(obj.id, IDMPolicy, obj, self.generate_seed())

    def after_step(self, *args, **kwargs) -> None:  # noqa: ANN003, ARG002, ANN002
        """Update the state of all spawned objects."""
        for obj in self.spawned_objects.values():
            obj.after_step()


class AdversaryMetaDriveEnv(MetaDriveEnv):
    """Adversary MetaDrive Environment."""

    def setup_engine(self) -> None:
        """Set up the engine for MetaDrive."""
        super().setup_engine()
        self.engine.update_manager("traffic_manager", MovingExampleManager())  # replace existing traffic manager

    @classmethod
    def default_config(cls) -> dict:
        """Set default configuration for the environment."""
        cfg = super().default_config()
        cfg.update(
            dict(
                traffic_density=0.0,
                decision_repeat=5,
                crash_vehicle_done=False,
                out_of_road_done=True,
                map="SSS",
                num_idm_victims=5,
                victim_crash_reward=10.0,
                ego_crash_penalty=-10.0,
                forward_reward=0.05,
                speed_reward=0.1,
            ),
        )
        return cfg

    def reset(self, seed: int, options: Any = None) -> tuple:  # noqa: ARG002
        """Call the parent reset method."""
        obs, info = super().reset(seed)
        return obs, info

    def spawn_victims(self) -> None:
        """Spawn victim vehicles in the environment."""
        lane = self.agent.lane
        for k in range(self.config["num_idm_victims"]):
            # spawn victim vehicles
            obj = self.spawn_object(
                DefaultVehicle,
                vehicle_config=dict(),
                lane=lane,
                position=(-10 * k, 0),
                heading=0,
            )
            self.add_policy(obj.id, IDMPolicy, obj, self.generate_seed())

    def reward_function(self, vehicle_id: str) -> float:
        """Define reward function for adversary vehicles."""
        ego = self.agents[vehicle_id]
        step_info = dict()

        # ego crash penalty
        if ego.crash_vehicle or ego.crash_object or ego.crash_sidewalk or self._is_out_of_road(ego):
            step_info.update(ego_crashed=True, behind_crashes=0)
            return -self.config["ego_crash_penalty"], step_info

        behind_crashes = 0
        long_ego, _ = ego.navigation.current_ref_lanes[0].local_coordinates(ego.position)

        # victim crash reward
        for obj_id, obj in self.engine.get_objects().items():
            if obj_id == vehicle_id:
                continue
            long_v, _ = obj.lane.local_coordinates(obj.position)
            if long_v < long_ego - 1.0 and (obj.crash_vehicle or obj.crash_object or obj.crash_sidewalk):
                behind_crashes += 1

        # positive reward is linear in number of victim crashes this step
        sparse_reward = self.config["victim_crash_reward"] * behind_crashes
        step_info["behind_crashes"] = behind_crashes

        # dense reward for forward progress and speed
        long_last, _ = ego.lane.local_coordinates(ego.last_position)
        long_now, lateral_now = ego.lane.local_coordinates(ego.position)
        forward_progress = long_now - long_last

        speed_factor = ego.speed_km_h / ego.max_speed_km_h
        dense_reward = self.config["forward_reward"] * forward_progress + self.config["speed_reward"] * speed_factor

        reward = dense_reward + sparse_reward
        return reward, step_info


# %%

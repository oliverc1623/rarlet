from typing import Any

import numpy as np
from metadrive.component.vehicle.vehicle_type import DefaultVehicle
from metadrive.envs import MetaDriveEnv
from metadrive.manager.traffic_manager import TrafficManager
from metadrive.policy.expert_policy import ExpertPolicy
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.scenario.utils import get_type_from_class


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


class MovingExampleManager(TrafficManager):
    """A custom manager for the MetaDrive environment."""

    def __init__(self, seed: int = 0, expert_vehicle_ratio: float = 0.5):
        super().__init__()
        self.init_velo = self.engine.global_config["init_velo"]
        self.rseed = seed
        self.expert_vehicle_ratio = clip(expert_vehicle_ratio, 0.0, 1.0)

    def _create_basic_vehicles(self, map, traffic_density) -> None:  # noqa: A002, ANN001, ARG002
        """Create basic vehicles for the environment with one protagonist agent."""
        total_num = len(self.respawn_lanes)
        for lane in self.respawn_lanes:
            _traffic_vehicles = []
            total_num = int(lane.length / self.VEHICLE_GAP)
            vehicle_longs = [i * self.VEHICLE_GAP for i in range(1, total_num + 1)]
            self.np_random.shuffle(vehicle_longs)
            v = vehicle_longs[: int(np.ceil(traffic_density * len(vehicle_longs)))]
            for long in v:
                vehicle_type = self.random_vehicle_type()
                policy = ExpertPolicy if self.np_random.random() < self.expert_vehicle_ratio else IDMPolicy
                traffic_v_config = {
                    "spawn_lane_index": lane.index,
                    "spawn_longitude": long,
                    "spawn_velocity": self.init_velo,
                    "use_special_color": False,
                }

                traffic_v_config.update(self.engine.global_config["traffic_vehicle_config"])
                random_v = self.spawn_object(vehicle_type, vehicle_config=traffic_v_config)
                self.add_policy(random_v.id, policy, random_v, self.rseed)
                self._traffic_vehicles.append(random_v)

        # Add the protagonist vehicle
        last_lane = self.respawn_lanes[-1]
        protagonist_v_config = {
            "spawn_lane_index": last_lane.index,
            "spawn_longitude": -10,
            "spawn_velocity": self.init_velo,
            "use_special_color": False,
        }
        protagonist = self.spawn_object(DefaultVehicle, vehicle_config=protagonist_v_config)
        self.add_policy(protagonist.id, ExpertPolicy, protagonist, self.rseed)
        self._traffic_vehicles.append(protagonist)


class AdversaryMetaDriveEnv(MetaDriveEnv):
    """Adversary MetaDrive Environment."""

    def setup_engine(self) -> None:
        """Set up the engine for MetaDrive."""
        super().setup_engine()
        self.engine.update_manager(
            "traffic_manager",
            MovingExampleManager(
                seed=self.config["start_seed"],
                expert_vehicle_ratio=self.config["expert_vehicle_ratio"],
            ),
        )

    @classmethod
    def default_config(cls) -> dict:
        """Set default configuration for the environment."""
        cfg = super().default_config()
        cfg.update(
            dict(
                traffic_density=0.0,
                decision_repeat=5,
                crash_vehicle_done=True,
                out_of_road_done=True,
                map="SSS",
                victim_crash_reward=20.0,
                ego_crash_penalty=10.0,
                forward_reward=0.5,
                speed_reward=0.75,
                living_penalty=0.5,
                brake_trigger_dist=10.0,
                k_brake=2.0,
                init_velo=(10, 0),
                truncate_as_terminate=True,
                expert_vehicle_ratio=0.5,
            ),
        )
        return cfg

    def reset(self, seed: int, options: Any = None) -> tuple:  # noqa: ARG002
        """Call the parent reset method."""
        obs, info = super().reset(seed)
        return obs, info

    def _distance(self, point1: tuple[float, float], point2: tuple[float, float]) -> float:
        """Calculate the longitudinal distance between two points."""
        return abs(point1[0] - point2[0])

    def reward_function(self, vehicle_id: str) -> float:
        """Define reward function for adversary vehicles."""
        ego = self.agents[vehicle_id]
        step_info = dict()

        # ego crash penalty
        if ego.crash_vehicle or ego.crash_object or ego.crash_sidewalk or self._is_out_of_road(ego):
            step_info.update(ego_crashed=True, behind_crashes=0, forward_reward=0.0, osc_reward=0.0)
            return -self.config["ego_crash_penalty"], step_info

        behind_crashes = 0

        # victim crash reward
        min_dist = float("inf")
        for obj_id, obj in self.engine.get_objects().items():
            if obj_id == ego.id:
                continue
            if (get_type_from_class(type(obj)) == "VEHICLE") and (obj.crash_vehicle or obj.crash_object or obj.crash_sidewalk):
                behind_crashes += 1
            if get_type_from_class(type(obj)) == "VEHICLE" and obj_id != ego.id and ego.lane == obj.lane:
                dist = self._distance(ego.position, obj.position)
                if dist < min_dist:
                    min_dist = dist

        # positive reward is linear in number of victim crashes this step
        if behind_crashes > 0:
            sparse_reward = self.config["victim_crash_reward"]
        else:
            sparse_reward = 0.0
        step_info["behind_crashes"] = behind_crashes

        done = self.done_function(vehicle_id)[0]
        if done and behind_crashes == 0:
            sparse_reward = -self.config["ego_crash_penalty"]

        reward = sparse_reward
        return reward, step_info

    def done_function(self, vehicle_id: str) -> tuple[bool, dict]:
        """Define done function for adversary vehicles."""
        super().done_function(vehicle_id)
        ego = self.agents[vehicle_id]
        done, info = super().done_function(vehicle_id)
        for obj_id, obj in self.engine.get_objects().items():
            if obj_id == ego.id:
                continue
            if (get_type_from_class(type(obj)) == "VEHICLE") and (obj.crash_vehicle or obj.crash_object or obj.crash_sidewalk):
                done = True
        return done, info


# %%

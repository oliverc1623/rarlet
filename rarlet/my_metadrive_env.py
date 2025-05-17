from typing import Any

import numpy as np
from metadrive.component.vehicle.vehicle_type import DefaultVehicle
from metadrive.envs import MetaDriveEnv
from metadrive.manager.traffic_manager import TrafficManager
from metadrive.policy.expert_policy import ExpertPolicy
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
                traffic_v_config = {"spawn_lane_index": lane.index, "spawn_longitude": long}
                traffic_v_config.update(self.engine.global_config["traffic_vehicle_config"])
                random_v = self.spawn_object(vehicle_type, vehicle_config=traffic_v_config)
                from metadrive.policy.idm_policy import IDMPolicy

                self.add_policy(random_v.id, IDMPolicy, random_v, self.generate_seed())
                self._traffic_vehicles.append(random_v)
        # Add the protagonist vehicle
        last_lane = self.respawn_lanes[-1]
        protagonist_v_config = {
            "spawn_lane_index": last_lane.index,
            "spawn_longitude": 0,
            "use_special_color": True,
            "spawn_velocity": (0, 0),
        }
        protagonist = self.spawn_object(DefaultVehicle, vehicle_config=protagonist_v_config)
        self.add_policy(protagonist.id, ExpertPolicy, protagonist, self.generate_seed())


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
                crash_vehicle_done=True,
                out_of_road_done=True,
                map="SSS",
                victim_crash_reward=10.0,
                ego_crash_penalty=10.0,
                forward_reward=1.0,
                speed_reward=1.0,
                living_penalty=0.1,
                brake_trigger_dist=10.0,
                k_brake=2.0,
            ),
        )
        return cfg

    def reset(self, seed: int, options: Any = None) -> tuple:  # noqa: ARG002
        """Call the parent reset method."""
        obs, info = super().reset(seed)
        return obs, info

    def _distance(self, point1: tuple[float, float], point2: tuple[float, float]) -> float:
        """Calculate the longitudinal distance between two points."""
        return point1[0] - point2[0]

    def reward_function(self, vehicle_id: str) -> float:
        """Define reward function for adversary vehicles."""
        ego = self.agents[vehicle_id]
        step_info = dict()

        # ego crash penalty
        if ego.crash_vehicle or ego.crash_object or ego.crash_sidewalk or self._is_out_of_road(ego):
            step_info.update(ego_crashed=True, behind_crashes=0)
            return -self.config["ego_crash_penalty"], step_info

        behind_crashes = 0

        # victim crash reward
        min_dist = float("inf")
        for obj_id, obj in self.engine.get_objects().items():
            if obj_id == vehicle_id:
                continue
            if (get_type_from_class(type(obj)) == "VEHICLE") and (obj.crash_vehicle or obj.crash_object or obj.crash_sidewalk):
                behind_crashes += 1
            if get_type_from_class(type(obj)) == "VEHICLE":
                dist = self._distance(ego.position, obj.position)
                if dist < min_dist:
                    min_dist = dist

        # positive reward is linear in number of victim crashes this step
        if behind_crashes > 0:
            sparse_reward = self.config["victim_crash_reward"]
        else:
            sparse_reward = 0.0
        step_info["behind_crashes"] = behind_crashes

        living_cost = -self.config["living_penalty"]

        # dense reward for forward progress
        long_last, _ = ego.lane.local_coordinates(ego.last_position)
        long_now, lateral_now = ego.lane.local_coordinates(ego.position)
        forward_progress = long_now - long_last
        forward_r = self.config["forward_reward"] * forward_progress

        # speed reward
        speed_factor = ego.speed_km_h / ego.max_speed_km_h
        speed_r = self.config["speed_reward"] * speed_factor

        # brake reward
        v_now = ego.speed
        v_last = ego.last_speed
        speed_diff = v_now - v_last
        a_long = speed_diff / 0.02

        near = min_dist < self.config["brake_trigger_dist"]
        brake_r = self.config["k_brake"] * max(0.0, -a_long) * near

        dense_reward = forward_r + speed_r + living_cost + brake_r

        reward = dense_reward + sparse_reward
        return reward, step_info

    def done_function(self, vehicle_id: str) -> tuple[bool, dict]:
        """Define done function for adversary vehicles."""
        super().done_function(vehicle_id)
        done, info = super().done_function(vehicle_id)
        for obj_id, obj in self.engine.get_objects().items():
            if obj_id == vehicle_id:
                continue
            if (get_type_from_class(type(obj)) == "VEHICLE") and (obj.crash_vehicle or obj.crash_object or obj.crash_sidewalk):
                done = True
        return done, info


# %%

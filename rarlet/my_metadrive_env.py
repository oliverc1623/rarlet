from typing import Any

from metadrive.envs import MetaDriveEnv


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


class AdversaryMetaDriveEnv(MetaDriveEnv):
    """Adversary MetaDrive Environment."""

    def __init__(self, config: dict):
        super().__init__(config)

    def reset(self, seed: int, options: Any = None) -> tuple:  # noqa: ARG002
        """Call the parent reset method."""
        obs, info = super().reset(seed)
        return obs, info

    def reward_function(self, vehicle_id: str) -> float:
        """Define reward function for adversary vehicles."""
        vehicle = self.agents[vehicle_id]
        step_info = dict()

        # Reward for moving forward in current lane
        if vehicle.lane in vehicle.navigation.current_ref_lanes:
            current_lane = vehicle.lane
            positive_road = 1
        else:
            current_lane = vehicle.navigation.current_ref_lanes[0]
            current_road = vehicle.navigation.current_road
            positive_road = 1 if not current_road.is_negative_road() else -1
        long_last, _ = current_lane.local_coordinates(vehicle.last_position)
        long_now, lateral_now = current_lane.local_coordinates(vehicle.position)

        # reward for lane keeping, without it vehicle can learn to overtake but fail to keep in lane
        if self.config["use_lateral_reward"]:
            lateral_factor = clip(1 - 2 * abs(lateral_now) / vehicle.navigation.get_current_lane_width(), 0.0, 1.0)
        else:
            lateral_factor = 1.0

        reward = 0.0
        reward += self.config["driving_reward"] * (long_now - long_last) * lateral_factor * positive_road
        reward += self.config["speed_reward"] * (vehicle.speed_km_h / vehicle.max_speed_km_h) * positive_road

        step_info["step_reward"] = reward

        if self._is_arrive_destination(vehicle):
            reward = +self.config["success_reward"]
        elif self._is_out_of_road(vehicle):
            reward = -self.config["out_of_road_penalty"]
        elif vehicle.crash_vehicle:
            reward = -self.config["crash_vehicle_penalty"]
        elif vehicle.crash_object:
            reward = -self.config["crash_object_penalty"]
        elif vehicle.crash_sidewalk:
            reward = -self.config["crash_sidewalk_penalty"]
        step_info["route_completion"] = vehicle.navigation.route_completion

        return reward, step_info

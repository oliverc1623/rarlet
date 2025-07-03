#SET MAP AND MODEL (i.e. definitions of all referenceable vehicle types, road library, etc)
# Imports
import math
import numpy as np

param map = localPath('../maps/Town06.xodr')
param carla_map = 'Town06'
param time_step = 1.0/10
model scenic.simulators.metadrive.model
param verifaiSamplerType = 'halton' # TODO: use scenic/random/uniform/halton sampler to train from scratch; then use ce for fine-tuning

monitor Reaches(obj1, region):
	while obj1 not in region:
		reward = 0.0
		long_now = obj1.position[0]
		long_last = obj1.last_position[0]
		distance = long_now - long_last
		speed_reward = obj1.speed / obj1.max_speed_mps
		reward += distance * 1.0
		reward += speed_reward * 0.1
		obj1.last_position = obj1.position
		obj1.reward = reward
		wait
	obj1.reward = 10.0  # Reward for reaching the goal region
	wait  # Wait for a moment before terminating
	terminate  # Terminate the simulation when the goal is reached

monitor StayInLane(obj1):
	while True:
		wait
		if not obj1._lane:
			obj1.reward = -20.0
			wait
			terminate

#PLACEMENT

goal_region = RectangularRegion((200,-146.5,0), 0, 60, 60)
ego = new Car at (100, VerifaiRange(-150.5, -136.5))

require monitor Reaches(ego, goal_region)
require monitor StayInLane(ego)

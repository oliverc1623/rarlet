#SET MAP AND MODEL (i.e. definitions of all referenceable vehicle types, road library, etc)
# Imports
import math
import numpy as np

param map = localPath('../maps/Town06.xodr')
param carla_map = 'Town06'
param time_step = 1.0/10
model scenic.simulators.metadrive.model
param verifaiSamplerType = 'halton' # TODO: use scenic/random/uniform/halton sampler to train from scratch; then use ce for fine-tuning

behavior dummy_attacker():
	while True:
		take SetThrottleAction(1.0), SetBrakeAction(0.0), SetSteerAction(0.0)

monitor Reaches(obj1, region):
    reached = False
    while not reached:
        if obj1 in region:
            obj1.reward = 10.0
            break
        else:
            if obj1.last_position is not None:
                long_now = obj1.position[0]
                long_last = obj1.last_position[0]
                distance = long_now - long_last
                obj1.reward = distance
            obj1.last_position = obj1.position
        wait
    terminate

monitor StayInLane(obj1):
    while True:
        if not obj1._lane:
            obj1.reward = -5.0
            break
        wait
    terminate

#PLACEMENT
ego_spawn_pt  = (100 @ -146.5)

goal_region = RectangularRegion((250,-146.5,0), 0, 20, 20)

num_vehicles_to_place = 4
lane_width = 3.5

id = 0
ego = new Car on ego_spawn_pt, with behavior dummy_attacker()

require monitor Reaches(ego, goal_region)
require monitor StayInLane(ego)

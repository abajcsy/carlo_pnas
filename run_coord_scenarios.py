from __future__ import division
import numpy as np
from enum import IntEnum
import random
import matplotlib.pyplot as plt
import matplotlib
from multi_agent_mdp import MultiAgentMDP

# construct dimensions of the world and start/goal locations
XA = 10
YA = 10
XB = 10
YB = 10
startA = [7,3]#[0,4]
startB = [7,6]#[2,7]
goalA = [3,6]#[5,9]
goalB = [3,3]#[1,5]
static_obstacles = [[5,6], [5,7], [5,8], [5,9], [5, 3], [5, 2], [5, 1], [5, 0]]
obstacles = None
hor = 4


# startA = [4,5]
# startB = [5,5]
# goalA = [5,0]
# goalB = [5,0]
# static_obstacles = [[0,9]]

# define the multi-agent MDP
mdp = MultiAgentMDP(XA, YA, XB, YB, startA, startB, goalA, goalB, obstacles, static_obstacles, coll_rad=0.5)

xinit = startA+startB

# game = "FBSB"
game = "OLSB"
# game = "OLNE"
xtraj, _, _ = mdp.coordinate(xinit, hor, game_type=game)

mdp.vis_solution(xtraj)
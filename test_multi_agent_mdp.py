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
startA = [5,1]
startB = [9,5]
goalA = [5,9]
goalB = [1,5]
obstacles = None

# define the multi-agent MDP
mdp = MultiAgentMDP(XA, YA, XB, YB, startA, startB, goalA, goalB, obstacles)

# test out state helper functions
s = mdp.coor_to_state(startA[0], startA[1], startB[0], startB[1])

print('linear state s', s)

xA, yA, xB, yB = mdp.state_to_coor(s)

print('multi-dimensional coor: ', xA, yA, xB, yB)

# visualize the initial setup
mdp.vis(startA[0], startA[1], startB[0], startB[1])

# forward simulate each agent taking and action, and print the reward. 
aA = 1
aB = 2
r = mdp.get_rewardA(startA+startB, aA, aB)

print('reward after each agent takes their action: ', r)

xprime, _ = mdp.transition_helper(startA+startB, aA, aB)

mdp.vis(xprime)

# Generate action sequence of length hor. 
hor = 3
action_sequences = mdp.get_all_action_seq(hor)
print('all action sequences: ', action_sequences)
print('# of all action sequences: ', len(action_sequences))
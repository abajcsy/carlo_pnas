import numpy as np
from world import World
from agents import Car, RingBuilding, CircleBuilding, Painting, Pedestrian, PointAgent
from geometry import Point
import time
from tkinter import *
from ol_ne_solver import OLNE_2player_solver, OLNE_quadratic_affine_cost_2player_solver
from fb_ne_solver import FBNE_2player_solver, FBNE_quadratic_affine_cost_2player_solver

import matplotlib.pyplot as plt

dt = 0.1 # time steps in terms of seconds. In other words, 1/dt is the FPS.
world_width = 12 # in meters
world_height = 12

# The world is 120 meters by 120 meters. ppm is the pixels per meter.
w = World(dt, width = world_width, height = world_height, ppm = 6) 

# A PointAgent object is a dynamic object -- it can move. We construct it using its center location and heading angle (heading is unused).
xinitA = [6-2., 6+2., 1.4, -1.]                        # xinit = [px, py, vx, vy]
c1 = PointAgent(Point(xinitA[0],xinitA[1]), np.pi/2)
c1.max_speed = 30.0                                 # let's say the maximum is 30 m/s (108 km/h)
c1.velocity = Point(xinitA[2], xinitA[3])
w.add(c1)

# Construct the second agent
xinitB = [6+2., 6+2., -0.5, -4.]
c2 = PointAgent(Point(xinitB[0],xinitB[1]), np.pi/2, color='blue')
c2.max_speed = 30.0 # let's say the maximum is 30 m/s (108 km/h)
c2.velocity = Point(xinitB[2], xinitB[3])
w.add(c2)

# # Plan for the second agent! Setup goals for each agent
# gA = [10., 60.] 
# gB = [50., 10.]

# # This is just for visualization! We visualize the two agent's goals here. 
# goalAvis = PointAgent(Point(gA[0],gA[1]), np.pi/2, color='red', radius=0.8)
# goalBvis = PointAgent(Point(gB[0],gB[1]), np.pi/2, color='blue', radius=0.8)
# w.add(goalAvis)
# w.add(goalBvis)

# a33 = 0.0; a44 = 0.0 # set the inertia value
# b33 = 0.0; b44 = 0.0
# c = 1.     # scale penalty on Q, q
# d = 100.    # scale penalty on R
# A = np.array([[1, 0, 1, 0, 0, 0, 0, 0], 
#                 [0, 1, 0, 1, 0, 0, 0, 0], 
#                 [0, 0, a33, 0, 0, 0, 0, 0], 
#                 [0, 0, 0, a44, 0, 0, 0, 0], 
#                 [0, 0, 0, 0, 1, 0, 1, 0], 
#                 [0, 0, 0, 0, 0, 1, 0, 1], 
#                 [0, 0, 0, 0, 0, 0, b33, 0], 
#                 [0, 0, 0, 0, 0, 0, 0, b44]])

# B1 = np.zeros((8,2))
# B2 = np.zeros((8,2))
# B1[2,0] = 1.; B1[3,1] = 1.
# B2[6,0] = 1.; B2[7,1] = 1.
# Q1 = c*np.eye(8)
# Q2 = c*np.eye(8)
# R1 = d*np.eye(2)
# R2 = d*np.eye(2)
#-----------------------------------------------------------------------+ Below is the LQ example
dt = 0.1 
# We define a few parameters below:
A_individual = np.eye(4) + dt*np.concatenate((np.zeros((4,2)), np.array([[1,0],[0,1],[1,0],[0,1]])), 1)
B_individual = np.concatenate((np.zeros((2,2)), np.eye(2)), 0)
collision_avoidance = np.array([[1,0,-1,0], [0,0,0,0], [-1,0,1,0], [0,0,0,0]])
cost_1 = np.diag((1,20,2,20))
cost_2 = np.diag((2,20,1,20)) - collision_avoidance

# We consider the following dynamics:
A = np.kron(np.eye(2), A_individual)
B1 = dt*np.kron(np.array([[1],[0]]), B_individual)
B2 = dt*np.kron(np.array([[0],[1]]), B_individual)

# We define the cost matrices as follows:
Q1 = np.kron(cost_1, np.eye(2))
Q2 = np.kron(cost_2,np.eye(2))
R1 = np.eye(2)
R2 = np.eye(2)

q1 = np.kron(np.array([[-1.],[0.],[0.],[0.]]), np.array([[3.],[-2.]]))
q2 = np.kron(np.array([[0.],[0.],[-1.],[0.]]), np.array([[-2.],[-1.5]]))
x0 = np.concatenate((xinitA,xinitB)).reshape(-1,1)
#-----------------------------------------------------------------------+ Above is the LQ example
# xg = np.array(gA + [0,0] + gB + [0,0]).reshape(-1,1)
# x0 = np.array(xinitA + xinitB).reshape(-1,1)
xg = np.array([6,6,0,0,6,6,0,0]).reshape(-1,1)
T = 90

# Solve a two-player game to plan for agent 2.
info_pattern = "OL"

if info_pattern == "OL":
    x_traj, u1_traj, u2_traj, J1, J2 = OLNE_quadratic_affine_cost_2player_solver(A,B1,B2,Q1,Q2,R1,R2,T,x0-xg,q1,q2)
elif info_pattern == "FB":
    x_traj, u1_traj, u2_traj, J1, J2 = FBNE_quadratic_affine_cost_2player_solver(A,B1,B2,Q1,Q2,R1,R2,T,x0-xg,q1,q2)
else:
    print('This type of information pattern is not implemented!')
    import sys
    sys.exit(0)
# import pdb; pdb.set_trace()
print('J1: ' + str(J1))
print('J2: ' + str(J2))
print('agent A x-traj: ' + str(x_traj[0,:]))
print('agent A y-traj: ' + str(x_traj[1,:]))

# Plot the state trajectories 
plt.figure(0)
plt.plot(xinitA[0], xinitA[1], 'ro')
plt.plot(xinitB[0], xinitB[1], 'bo')
plt.plot(x_traj[0,:]+xg[0], x_traj[1,:]+xg[1], 'r-o')
plt.plot(x_traj[4,:]+xg[4], x_traj[5,:]+xg[5], 'b-o')
# plt.plot(gA[0], gA[1], 'ro')
# plt.plot(gB[0], gB[1], 'bo')
plt.axis([0, world_width, 0, world_height])
plt.title(info_pattern + ' LQGames planned state trajectories')
plt.show()

# Plot the control profile of agent 1
plt.figure(1)
plt.plot(np.arange(0,T), u1_traj[0,:], 'r')
plt.plot(np.arange(0,T), u1_traj[1,:], 'm')
plt.xlabel('tstep')
plt.ylabel('Agent 1 Control (x-accel: red, y-accel: magenta)')
plt.title('Planned control trajectory for Agent 1')
plt.show()

# Plot the control profile of agent 2
plt.figure(2)
plt.plot(np.arange(0,T), u2_traj[0,:], 'b')
plt.plot(np.arange(0,T), u2_traj[1,:], 'c')
plt.xlabel('tstep')
plt.ylabel('Agent 2 Control (x-accel: blue, y-accel: cyan)')
plt.title('Planned control trajectory for Agent 2')
plt.show()

w.render() # This visualizes the world we just constructed.

# Let's use the keyboard input for human control
from interactive_controllers import KeyboardController
c1.set_control(0., 0.) # Initially, the car will have 0 steering and 0 throttle.
controller = KeyboardController(w)

for k in range(T):
    # human-driven car controls come from keyboard
    # c1.set_control(controller.steering, controller.throttle)
    c1.set_control(-u1_traj[0,k], u1_traj[1,k]) 

    # robot controls are something random right now
    c2.set_control(-u2_traj[0,k], u2_traj[1,k]) 
    # print('agent 1 center: ' + str(c1.center))

    w.tick() # This ticks the world for one time step (dt second)
    w.render()
    time.sleep(dt) # if we do dt/4, then let's watch it 4x
    # if w.collision_exists():
    #     import sys
    #     sys.exit(0)
w.close()


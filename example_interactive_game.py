import numpy as np
from world import World
from agents import Car, RingBuilding, CircleBuilding, Painting, Pedestrian, PointAgent
from geometry import Point
import time
from tkinter import *
from ol_ne_solver import OLNE_2player_solver
from fb_ne_solver import FBNE_2player_solver

dt = 0.1 # time steps in terms of seconds. In other words, 1/dt is the FPS.
world_width = 120 # in meters
world_height = 120
inner_building_radius = 30
num_lanes = 1
lane_marker_width = 0.5
num_of_lane_markers = 50
lane_width = 3.5

# The world is 120 meters by 120 meters. ppm is the pixels per meter.
w = World(dt, width = world_width, height = world_height, ppm = 6) 

# Let's also add some lane markers on the ground. This is just decorative. Because, why not.
for lane_no in range(num_lanes - 1):
    lane_markers_radius = inner_building_radius + (lane_no + 1) * lane_width + (lane_no + 0.5) * lane_marker_width
    lane_marker_height = np.sqrt(2*(lane_markers_radius**2)*(1-np.cos((2*np.pi)/(2*num_of_lane_markers)))) # approximate the circle with a polygon and then use cosine theorem
    for theta in np.arange(0, 2*np.pi, 2*np.pi / num_of_lane_markers):
        dx = lane_markers_radius * np.cos(theta)
        dy = lane_markers_radius * np.sin(theta)
        w.add(Painting(Point(world_width/2 + dx, world_height/2 + dy), Point(lane_marker_width, lane_marker_height), 'white', heading = theta))

# A PointAgent object is a dynamic object -- it can move. We construct it using its center location and heading angle.
c1 = PointAgent(Point(91.75,60), np.pi/2)
c1.max_speed = 30.0 # let's say the maximum is 30 m/s (108 km/h)
c1.velocity = Point(0, 3.0)
w.add(c1)

# Construct the second agent
c2 = PointAgent(Point(10.,60), np.pi/2, color='blue')
c2.max_speed = 30.0 # let's say the maximum is 30 m/s (108 km/h)
c2.velocity = Point(0, 3.0)
w.add(c2)

# Plan for the second agent!
gA = [10., 60.] # TODO: make the agents go to the goal! 
gB = [90., 60.]
a33 = 0.9; a44 = 0.9
b33 = 0.9; b44 = 0.9
A = np.array([[1, 0, 1, 0, 0, 0, 0, 0], 
                [0, 1, 0, 1, 0, 0, 0, 0], 
                [0, 0, a33, 0, 0, 0, 0, 0], 
                [0, 0, 0, a44, 0, 0, 0, 0], 
                [0, 0, 0, 0, 1, 0, 1, 0], 
                [0, 0, 0, 0, 0, 1, 0, 1], 
                [0, 0, 0, 0, 0, 0, b33, 0], 
                [0, 0, 0, 0, 0, 0, 0, b44]])
B1 = np.zeros((8,2))
B2 = np.zeros((8,2))
B1[2,0] = 1.; B1[3,1] = 1.
B2[6,0] = 1.; B2[7,1] = 1.
Q1 = np.eye(8)
Q2 = np.eye(8)
R1 = np.eye(2)
R2 = np.eye(2)
x0 = np.array([91.75,60,0, 3.0,10.,60,0,3.0]).reshape(-1,1)
T = 600

# Solve a two-player game to plan for agent 2.
# TODO: Need to modify solver to have affine terms.

# x_traj, u1_traj, u2_traj, J1, J2 = OLNE_2player_solver(A,B1,B2,Q1,Q2,R1,R2,T,x0)
x_traj, u1_traj, u2_traj, J1, J2 = FBNE_2player_solver(A,B1,B2,Q1,Q2,R1,R2,T,x0)

w.render() # This visualizes the world we just constructed.

# Let's use the keyboard input for human control
from interactive_controllers import KeyboardController
c1.set_control(0., 0.) # Initially, the car will have 0 steering and 0 throttle.
controller = KeyboardController(w)

for k in range(T):
    # human-driven car controls come from keyboard
    c1.set_control(controller.steering, controller.throttle)
    # c1.set_control(u1_traj[0,k], u1_traj[1,k]) 

    # robot controls are something random right now
    c2.set_control(u2_traj[0,k], u2_traj[1,k]) 

    w.tick() # This ticks the world for one time step (dt second)
    w.render()
    time.sleep(dt/4) # Let's watch it 4x
    if w.collision_exists():
        import sys
        sys.exit(0)
w.close()


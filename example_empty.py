import numpy as np
from world import World
from agents import Car, RingBuilding, CircleBuilding, Painting, Pedestrian
from geometry import Point
import time
from tkinter import *

# human_controller = True

dt = 0.1 # time steps in terms of seconds. In other words, 1/dt is the FPS.
world_width = 120 # in meters
world_height = 120
inner_building_radius = 30
num_lanes = 1
lane_marker_width = 0.5
num_of_lane_markers = 50
lane_width = 3.5

w = World(dt, width = world_width, height = world_height, ppm = 6) # The world is 120 meters by 120 meters. ppm is the pixels per meter.


# Let's also add some lane markers on the ground. This is just decorative. Because, why not.
for lane_no in range(num_lanes - 1):
    lane_markers_radius = inner_building_radius + (lane_no + 1) * lane_width + (lane_no + 0.5) * lane_marker_width
    lane_marker_height = np.sqrt(2*(lane_markers_radius**2)*(1-np.cos((2*np.pi)/(2*num_of_lane_markers)))) # approximate the circle with a polygon and then use cosine theorem
    for theta in np.arange(0, 2*np.pi, 2*np.pi / num_of_lane_markers):
        dx = lane_markers_radius * np.cos(theta)
        dy = lane_markers_radius * np.sin(theta)
        w.add(Painting(Point(world_width/2 + dx, world_height/2 + dy), Point(lane_marker_width, lane_marker_height), 'white', heading = theta))
    

# A Car object is a dynamic object -- it can move. We construct it using its center location and heading angle.
c1 = Car(Point(91.75,60), np.pi/2)
c1.max_speed = 30.0 # let's say the maximum is 30 m/s (108 km/h)
c1.velocity = Point(0, 3.0)
w.add(c1)

# A Car object is a dynamic object -- it can move. We construct it using its center location and heading angle.
c2 = Car(Point(world_width/2.,60), np.pi/2, color='blue')
c2.max_speed = 30.0 # let's say the maximum is 30 m/s (108 km/h)
c2.velocity = Point(0, 3.0)
w.add(c2)

w.render() # This visualizes the world we just constructed.

# Let's use the keyboard input for human control
from interactive_controllers import KeyboardController
c1.set_control(0., 0.) # Initially, the car will have 0 steering and 0 throttle.
controller = KeyboardController(w)

for k in range(600):
    # human-driven car controls come from keyboard
    c1.set_control(controller.steering, controller.throttle)

    # robot controls are something random right now
    c2.set_control(np.random.rand(), np.random.rand()) 

    w.tick() # This ticks the world for one time step (dt second)
    w.render()
    time.sleep(dt/4) # Let's watch it 4x
    if w.collision_exists():
        import sys
        sys.exit(0)
w.close()
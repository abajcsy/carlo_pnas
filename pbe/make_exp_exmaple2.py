import numpy as np

# Combine controls
def combine(u1,u2):
    return np.vstack([u1, u2])

# Get individual agent's states
def decompose_dim(x, dim):
    return x[0:dim,:], x[dim:2*dim,:], x[2*dim:,:]

def vec(*kwargs):
    return np.array([list(kwargs)]).T

def project_belief(b, beliefs):
    # Project belief onto belief space
    return beliefs[np.argmin([np.linalg.norm(b-b1) for b1 in beliefs])]

def make_1d_exp(x_max=7, b_num=10):
    def dynamics(x,u1,u2):
        u = combine(u1,u2)
        x_next = x + u
        if any(x_next < 0) or any(x_next >= x_max): # Goes outside grid
            return x 
        return x_next

    def r1(x,u1,u2,theta):
        goal = vec(0)
        x1,x2,_ = decompose_dim(x,1)
        if theta == 1: # Go to other agent
            return -np.linalg.norm(x2-6)**2 - np.linalg.norm(u1)**2
        else: # Go to goal
            return -np.linalg.norm(x1-0)**2 - np.linalg.norm(u1)**2
        
    def r2(x,u1,u2,theta):
        goal = vec(x_max)
        x1,x2,_ = decompose_dim(x,1)
        if theta == 1: # Go to other agent
            return -np.linalg.norm(x1-x2)**2 - np.linalg.norm(u2)**2
        else: # Go to goal
            return -np.linalg.norm(x2-6)**2 - np.linalg.norm(u2)**2
        
    # Setup states, actions, types
    states = []
    beliefs  = []
    for x1 in range(x_max):
        for x2 in range(x_max):
            for b11 in np.linspace(0,1,b_num):
                for b21 in np.linspace(0,1,b_num):
                    for b12 in np.linspace(0,1,b_num):
                        if b11+b21+b12 > 1:
                            continue # Skip invalid belief
                        states.append(vec(x1,x2,b11,b21,b12))
                        b = vec(b11,b21,b12)
                        if not np.any([np.all(b == vec(b11,b21,b12)) for b in beliefs]):
                            beliefs.append(vec(b11,b21,b12))

    actions = []
    for x in [-1,0,1]:
            actions.append(vec(x))
                                
    types = [1,2]

    return states, actions, types, dynamics, r1, r2, beliefs

def make_2d_exp(x_max=2, y_max=2, b_num=2):
    # 8x8 grid dynamics

    def dynamics(x,u1,u2):
        u = combine(u1,u2)
        x_next = x + u
        if any(x_next < 0) or any(x_next >= x_max): # Goes outside grid
            return x 
        return x_next

    def r1(x,u1,u2,theta):
        goal = vec(0,0)
        x1,x2,_ = decompose_dim(x,2)
        if theta == 1: # Go to other agent
            return -np.linalg.norm(x2-6)**2 - np.linalg.norm(u1)**2
        else: # Go to goal
            return -np.linalg.norm(x1-goal) - np.linalg.norm(u1)**2
        
    def r2(x,u1,u2,theta):
        goal = vec(x_max,y_max)
        x1,x2,_ = decompose_dim(x,2)
        if theta == 1: # Go to other agent
            return -np.linalg.norm(x1-x2) - np.linalg.norm(u2)
        else: # Go to goal
            return -np.linalg.norm(x2-goal) - np.linalg.norm(u2)
        
    # Setup states, actions, types
    states = []
    beliefs = []
    for x1 in range(x_max):
        for x2 in range(x_max):
            for y1 in range(y_max):
                for y2 in range(y_max):
                    for b11 in np.linspace(0,1,b_num):
                        for b21 in np.linspace(0,1,b_num):
                            for b12 in np.linspace(0,1,b_num):
                                if b11+b21+b12 > 1:
                                    continue # Skip invalid belief
                                states.append(vec(x1,x2,y1,y2,b11,b21,b12))
                                if not np.any([np.all(b == vec(b11,b21,b12)) for b in beliefs]):
                                    beliefs.append(vec(b11,b21,b12))

    actions = []
    for x in [-1,0,1]:
        for y in [-1,0,1]:
            actions.append(vec(x,y))
                                
    types = [1,2]

    return states, actions, types, dynamics, r1, r2, beliefs


import numpy as np
import matplotlib.pyplot as plt
import time
import pickle

# 8x8 grid dynamics

# Combine controls
def combine(u1,u2):
    return np.vstack([u1, u2])

# Get individual agent's states
def decompose(x):
    return x[0:2,0], x[2:4,0], x[4:,0]

def vec(*kwargs):
    return np.array([list(kwargs)]).T

def dynamics(x,u1,u2):
    u = combine(u1,u2)
    x_next = x + u
    if any(x_next <= 0) or any(x_next >=7): # Goes outside grid
        return x 
    return x_next

def r1(x,u1,u2,theta):
    goal = vec(0,0)
    x1,x2,_ = decompose(x)
    if theta == 1: # Go to other agent
        return -np.linalg.norm(x1-x2) - np.linalg.norm(u1)
    else: # Go to goal
        return -np.linalg.norm(x1-goal) - np.linalg.norm(u1)
    
def r2(x,u1,u2,theta):
    goal = vec(7,7)
    x1,x2,_ = decompose(x)
    if theta == 1: # Go to other agent
        return -np.linalg.norm(x1-x2) - np.linalg.norm(u1)
    else: # Go to goal
        return -np.linalg.norm(x1-goal) - np.linalg.norm(u1)
    
# Setup states, actions, types
states = []
for x1 in range(8):
    for x2 in range(8):
        for y1 in range(8):
            for y2 in range(8):
                for b11 in np.linspace(0,1,10):
                    for b21 in np.linspace(0,1,10):
                        for b12 in np.linspace(0,1,10):
                            states.append(vec(x1,x2,y1,y2,b11,b21,b12))

actions = []
for x in [-1,0,1]:
    for y in [-1,0,1]:
        actions.append(vec(x,y))
                            
types = [1,2]

print(f"Number of states = {len(states)}")

# Setup belief, value function updates
def belief_update(strat1,strat2,b,u1,u2):
    b11, b21, b12 = b[0,0], b[1,0], b[2,0]
    b22 = 1 - np.sum(b)
    p_t1_is_1 = b11 + b12
    p_t1_is_2 = 1 - p_t1_is_1
    p_t2_is_1 = b11 + b21
    p_t2_is_2 = 1 - p_t2_is_1
    
    one = lambda x,y: (1 if x==y else 0)
    denom1 = one(u1,strat1[1])*p_t1_is_1 + one(u1,strat1[2])*p_t1_is_2
    denom2 = one(u2,strat2[1])*p_t2_is_1 + one(u2,strat2[2])*p_t2_is_2
    denom = denom1*denom2
    b11_next = one(u1,strat1[1])*one(u2,strat2[1])*p_t1_is_1*p_t2_is_1/denom
    b21_next = one(u1,strat1[2])*one(u2,strat2[1])*p_t1_is_2*p_t2_is_1/denom
    b12_next = one(u1,strat1[1])*one(u2,strat2[2])*p_t1_is_1*p_t2_is_2/denom
    
    return vec(b11_next, b21_next, b12_next)

def calc_v_next(u1,u2,v_nexts,state,strat1,strat2,t):
    x1,x2,b = decompose(state)
    x = combine(x1,x2)
    x_next = dynamics(x, u1, u2)
    b_next = belief_update(strat1, strat2, b, u1, u2)
    state_next = combine(x_next, b_next)
    return v_nexts[(arr_tup(state_next),t)]

def arr_tup(x):
    return tuple(x.flatten())



# Setup value function and policy dictionaries

T = 10 # horizon
def make_state_type_list():
    return {(arr_tup(state),t):None for state in states for t in types}
def make_state_list():
    return {(arr_tup(state)):None for state in states}

# Terminal state
V_1 = {} # value functions: [t][(state,type)] = value
V_1[10] = {(arr_tup(state),t):0 for state in states for t in types}
V_2 = {}
V_2[10] = {(arr_tup(state),t):0 for state in states for t in types}

pi_1, pi_2 = {}, {} # policies: [t][state][type] -> action


# Run PBE computation
for t in np.flip(range(10)):
    print(f"t = {t}")
    V_1_t = make_state_type_list()
    V_2_t = make_state_type_list()
    pi_1_t, pi_2_t = make_state_list(), make_state_list()
    i = 0
    start = time.time()
    for state in states:
        # Print progress in percent
        start_i = time.time()
        if i % 40960 == 0:
            print(f"{i/len(states)*100:.2f}%")
        i += 1
        x1,x2,b = decompose(state)
        b11, b21, b12 = b[0,0], b[1,0], b[2,0]
        b22 = 1 - np.sum(b)

        # Get list of all strategies
        strats1, strats2 = [], []
        for strats in [strats1, strats2]:
            for a in actions:
                for b in actions:
                    strats.append({types[1]:a, types[2]:b})

        # Calculate utility of each strategy pair for different types
        us1, us2 = {(s1,s2):{} for s1 in strats1 for s2 in strats2}, {(s1,s2):{} for s1 in strats1 for s2 in strats2}
        for t1 in types:
            for s1 in strats1:
                for s2 in strats2:
                    if t1 == 1:
                        p1 = b11/(b11+b12)
                        p2 = 1 - p1
                        v1 = calc_v_next(s1[1], s2[1],V_1[t+1], state, s1, s2, t1)
                        v2 = calc_v_next(s1[1], s2[2],V_1[t+1], state, s1, s2, t1)
                        u1 = p1*(r1(x,s1[1],s2[1],1)+ v1) + p2*(r1(x,s1[1],s2[2],1) + v2)
                    else:
                        p1 = b21/(b21+b22)
                        p2 = 1-p1
                        v1 = calc_v_next(s1[2], s2[1],V_1[t+1], state, s1, s2, t1)
                        v2 = calc_v_next(s1[2], s2[2],V_1[t+1], state, s1, s2, t1)
                        u1 = p1*(r1(x,s1[1],s2[1],2) + v1) + p2*(r1(x,s1[1],s2[2],2) + v2)
                    us1[t1][(s1,s2)] = u1
        for t2 in types:
            for s1 in strats1:
                for s2 in strats2:
                    if t2 == 1:
                        p1 = b11/(b12+b22)
                        p2 = 1 - p1
                        v1 = calc_v_next(s1[1], s2[1],V_2[t+1], state, s1, s2, t2)
                        v2 = calc_v_next(s1[2], s2[1],V_2[t+1], state, s1, s2, t2)
                        u2 = p1*(r2(x,s1[1],s2[1],1)+v1) + p2*(r1(x,s1[2],s2[1],1)+v2)
                    else:
                        p1 = b12/(b12+b22)
                        p2 = 1 - p1
                        v1 = calc_v_next(s1[1], s2[2],V_2[t+1], state, s1, s2, t2)
                        v2 = calc_v_next(s1[2], s2[2],V_2[t+1], state, s1, s2, t2)
                        u2 = p1*(r2(x,s1[1],s2[1],2) + v1) + p2*(r2(x,s1[1],s2[2],2) + v2)
                    us2[t2][(s1,s2)] = u2

        # Find equilibria
        eqs = []
        for s1 in strats1:
            for s2 in strats2:
                not_eq = False
                # Check player 1s condition
                for t1 in types:
                    for s1_prime in strats1:
                        if us1[t1][(s1,s2)] < us1[t1][(s1_prime,s2)]:
                            not_eq = True
                            break
                    if not_eq:
                        break
                if not_eq:
                    continue
                # Check player 2s condition
                for t2 in types:
                    for s2_prime in strats2:
                        if us2[t2][(s1,s2)] < us2[t2][(s1,s2_prime)]:
                            not_eq = True
                            break
                    if not_eq:
                        break
                
                if not not_eq:
                    eqs.append((s1,s2))
        
        # Equilibrium selection
        if len(eqs) == 0:
            print("No equilibrium found at state", state, "at time", t)
            for t in types:
                V_1_t[(state,t)] = -1000
                V_2_t[(state,t)] = -1000
                pi_1_t[state] = None
                pi_2_t[state] = None
        elif len(eqs) == 1:
            eq = eqs[0]
        else:
            eq = eqs[0]
        
        pi_1_t[state] = eq[0]
        pi_2_t[state] = eq[1]

        # Calculate value of state
        for t in types:
            V_1_t[(state,t)] = us1[t][eq]
            V_2_t[(state,t)] = us2[t][eq]

        print("Time elapsed for single state:", time.time()-start)
    
    V_1[t] = V_1_t
    V_2[t] = V_2_t
    pi_1[t] = pi_1_t
    pi_2[t] = pi_2_t
    end = time.time()
    print(f"Time for iteration {t}: {end-start:.2f}s")

# Save data
with open(f"V_1.pkl", "wb") as f:
    pickle.dump(V_1, f)
with open(f"V_2.pkl", "wb") as f:
    pickle.dump(V_2_t, f)
with open(f"pi_1.pkl", "wb") as f:
    pickle.dump(pi_1, f)
with open(f"pi_2.pkl", "wb") as f:
    pickle.dump(pi_2, f)

    
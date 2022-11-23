import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
from multiprocessing.pool import ThreadPool as Pool
from make_exp import *
import argparse

# Setup arg parser
# Example: python3 soln_pbe.py --env 1d --size 7 --beliefs 10
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default="1d")
parser.add_argument('--size', type=int, default=5)
parser.add_argument('--beliefs', type=int, default=10)
parser.add_argument('--multi', action="store_true", default=False)
args = parser.parse_args()

multi = args.multi

x_size = args.size
b_num = args.beliefs

if args.env == "1d":
    states, actions, types, dynamics, r1, r2, beliefs = make_1d_exp(x_size, b_num)
    decompose = lambda x: decompose_dim(x,1)
elif args.env == "2d":
    states, actions, types, dynamics, r1, r2, beliefs = make_2d_exp(x_size, x_size, b_num)
    decompose = lambda x: decompose_dim(x,2)

print(f"Number of states = {len(states)}")

# Setup belief, value function updates
def belief_update(strat1,strat2,b,u1,u2):
    b11, b21, b12 = b[0,0], b[1,0], b[2,0]
    b22 = 1 - np.sum(b)
    p_t1_is_1 = b11 + b12
    p_t1_is_2 = 1 - p_t1_is_1
    p_t2_is_1 = b11 + b21
    p_t2_is_2 = 1 - p_t2_is_1
    
    one = lambda x,y: (1 if np.all(x==y) else 0)
    denom1 = one(u1,strat1[1])*p_t1_is_1 + one(u1,strat1[2])*p_t1_is_2
    denom2 = one(u2,strat2[1])*p_t2_is_1 + one(u2,strat2[2])*p_t2_is_2
    denom = denom1*denom2
    b11_next = one(u1,strat1[1])*one(u2,strat2[1])*p_t1_is_1*p_t2_is_1/denom
    b21_next = one(u1,strat1[2])*one(u2,strat2[1])*p_t1_is_2*p_t2_is_1/denom
    b12_next = one(u1,strat1[1])*one(u2,strat2[2])*p_t1_is_1*p_t2_is_2/denom
    b_next = vec(b11_next, b21_next, b12_next)
    return beliefs[np.argmin(np.sum(np.abs(beliefs-b_next),axis=1))]

def calc_v_next(u1,u2,v_nexts,state,strat1,strat2,t):
    x1,x2,b = decompose(state)
    x = combine(x1,x2)
    x_next = dynamics(x, u1, u2)
    b_next = belief_update(strat1, strat2, b, u1, u2)
    if np.any(np.isnan(b_next)):
        return -np.inf
    state_next = combine(x_next, b_next)
    try:
        res = v_nexts[(arr_tup(state_next),t)]
    except:
        print("Error")
        # Print variables
        print(f"u1 = {u1}")
        print(f"u2 = {u2}")
        print(f"state = {state}")
        print(f"strat1 = {strat1}")
        print(f"strat2 = {strat2}")
        print(f"t = {t}")
        print(f"state_next = {state_next.T}")
        print(f"b_next = {b_next.T}")
        raise Exception("Error")
    return res

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

# One state computation
def compute_state(state, tau, V_1_t, V_2_t, pi_1_t, pi_2_t, V_1_next, V_2_next):
    start_i = time.time()
    x1,x2,b = decompose(state)
    
    b11, b21, b12 = b[0,0], b[1,0], b[2,0]
    b22 = 1 - np.sum(b)

    # Get list of all strategies
    strats1, strats2 = [], []
    for strats in [strats1, strats2]:
        for a in actions:
            for b in actions:
                strats.append({types[0]:a, types[1]:b})

    # Calculate utility of each strategy pair for different types
    make_pairs = lambda : {(i1,i2):{} for i1 in range(len(strats1)) for i2 in range(len(strats2))}
    us1, us2 = {t:make_pairs() for t in types}, {t:make_pairs() for t in types}
    for t1 in types:
        for i1 in range(len(strats1)):
            for i2 in range(len(strats2)):
                s1, s2 = strats1[i1], strats2[i2]
                if t1 == 1:
                    p1 = b11/(b11+b12)
                    if np.isnan(p1):
                        u1 = -np.inf
                    else:
                        p2 = 1 - p1
                        v1 = calc_v_next(s1[1], s2[1],V_1_next, state, s1, s2, t1)
                        v2 = calc_v_next(s1[1], s2[2],V_1_next, state, s1, s2, t1)
                        u1 = p1*(r1(state,s1[1],s2[1],1)+ v1) + p2*(r1(state,s1[1],s2[2],1) + v2)
                else:
                    p1 = b21/(b21+b22)
                    if np.isnan(p1):
                        u1 = -np.inf
                    else:
                        p2 = 1-p1
                        v1 = calc_v_next(s1[2], s2[1],V_1_next, state, s1, s2, t1)
                        v2 = calc_v_next(s1[2], s2[2],V_1_next, state, s1, s2, t1)
                        u1 = p1*(r1(state,s1[1],s2[1],2) + v1) + p2*(r1(state,s1[1],s2[2],2) + v2)
                us1[t1][(i1,i2)] = u1
    for t2 in types:
        for i1 in range(len(strats1)):
            for i2 in range(len(strats2)):
                s1, s2 = strats1[i1], strats2[i2]
                if t2 == 1:
                    p1 = b11/(b12+b22)
                    if np.isnan(p1):
                        u2 = -np.inf
                    else:
                        p2 = 1 - p1
                        v1 = calc_v_next(s1[1], s2[1],V_2_next, state, s1, s2, t2)
                        v2 = calc_v_next(s1[2], s2[1],V_2_next, state, s1, s2, t2)
                        u2 = p1*(r2(state,s1[1],s2[1],1)+v1) + p2*(r2(state,s1[2],s2[1],1)+v2)
                else:
                    p1 = b12/(b12+b22)
                    if np.isnan(p1):
                        u2 = -np.inf
                    else:
                        p2 = 1 - p1
                        v1 = calc_v_next(s1[1], s2[2],V_2_next, state, s1, s2, t2)
                        v2 = calc_v_next(s1[2], s2[2],V_2_next, state, s1, s2, t2)
                        u2 = p1*(r2(state,s1[1],s2[1],2) + v1) + p2*(r2(state,s1[1],s2[2],2) + v2)
                us2[t2][(i1,i2)] = u2

    # Find equilibria
    eqs = []
    for i1 in range(len(strats1)):
        for i2 in range(len(strats2)):
            s1, s2 = strats1[i1], strats2[i2]
            not_eq = False
            # Check player 1s condition
            for t1 in types:
                for i1_prime in range(len(strats1)):
                    if us1[t1][(i1,i2)] < us1[t1][(i1_prime,i2)]:
                        not_eq = True
                        break
                if not_eq:
                    break
            if not_eq:
                continue
            # Check player 2s condition
            for t2 in types:
                for i2_prime in range(len(strats2)):
                    if us2[t2][(i1,i2)] < us2[t2][(i1,i2_prime)]:
                        not_eq = True
                        break
                if not_eq:
                    break
            
            if not not_eq:
                eqs.append((i1,i2))
    
    # Equilibrium selection
    if len(eqs) == 0:
        print("No equilibrium found at state", state, "at time", t)
        for t in types:
            V_1_t[(arr_tup(state),t)] = -1000
            V_2_t[(arr_tup(state),t)] = -1000
            pi_1_t[arr_tup(state)] = None
            pi_2_t[arr_tup(state)] = None
    elif len(eqs) == 1:
        eq = eqs[0]
    else:
        eq = eqs[0]
    
    pi_1_t[arr_tup(state)] = eq[0]
    pi_2_t[arr_tup(state)] = eq[1]

    # Calculate value of state
    for t in types:
        V_1_t[(arr_tup(state),t)] = us1[t][eq]
        V_2_t[(arr_tup(state),t)] = us2[t][eq]

    print("Time elapsed for single state:", time.time()-start_i)

# Run PBE computation
for tau in np.flip(range(10)):
    print(f"time = {tau}")
    V_1_t = make_state_type_list()
    V_2_t = make_state_type_list()
    pi_1_t, pi_2_t = make_state_list(), make_state_list()
    i = 0
    start = time.time()
    if multi:
        calc_state = lambda state: compute_state(state, tau, V_1_t, V_2_t, pi_1_t, pi_2_t, V_1[tau+1], V_2[tau+1])
        with Pool(16) as p:
            p.map(calc_state, states)
    else:
        for state in states:
            # Print progress in percent
            start_i = time.time()
            if i % 40960 == 0:
                print(f"{i/len(states)*100:.2f}%")
            i += 1
        
            compute_state(state, tau, V_1_t, V_2_t, pi_1_t, pi_2_t, V_1[tau+1], V_2[tau+1])
    
    V_1[tau] = V_1_t
    V_2[tau] = V_2_t
    pi_1[tau] = pi_1_t
    pi_2[tau] = pi_2_t
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
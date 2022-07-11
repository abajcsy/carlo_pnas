from __future__ import division
import numpy as np
from enum import IntEnum
import random
import matplotlib.pyplot as plt
import matplotlib

import itertools 

# Defining the multi-agent grid world.

class SingleAgentActions(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class MultiAgentMDP(object):
    """
    An X by Y MDP for a mutli-agent system in an environment with obstacles. 
    Reward is 
    modeled as a linear combination of features
    """
    SingleAgentActions = SingleAgentActions

    def __init__(self, XA, YA, XB, YB, startA, startB, goalA, goalB, obstacles):
        """
        Params:
            XA [int] -- agent A x-positions (the width of this gridworld).
            YA [int] -- agent A y-positions (the height of this gridworld).
            XB [int] -- agent B x-positions (the width of this gridworld).
            YB [int] -- agent B y-positions (the height of this gridworld).
            startA [tuple] -- Starting position for agent A specified in coords (x, y).
            startB [tuple] -- Starting position for agent B specified in coords (x, y).
            goalA [tuple] -- Goal position for agent A specified in coords (gx, gy).
            goalB [tuple] -- Goal position for agent B specified in coords (gx, gy).
            obstacles [list] -- List of axis-aligned 2D boxes that represent
                obstacles in the environment for the agent. Specified in coords:
                [[(lower_x, lower_y), (upper_x, upper_y)], [...]]
        """

        assert isinstance(XA, int), XA
        assert isinstance(YA, int), YA
        assert isinstance(XB, int), XB
        assert isinstance(YB, int), YB
        assert XA > 0
        assert YA > 0
        assert XB > 0
        assert YB > 0
        
        np.random.seed(1)

        # Set up variables for Multi-Agent MDP
        self.XA = XA
        self.YA = YA
        self.XB = XB
        self.YB = YB
        self.S = XA * YA * XB * YB 
        self.actionsA = len(SingleAgentActions)
        self.actionsB = len(SingleAgentActions)
        self.sA = startA
        self.sB = startB
        self.gA = goalA
        self.gB = goalB 
        self.start = self.coor_to_state(startA[0], startA[1], startB[0], startB[1])
        self.goal = self.coor_to_state(self.gA[0], self.gA[1], self.gB[0], self.gB[1])

        # Set the obstacles in the environment.
        self.obstacles = obstacles

    ###########################
    #### Planning functions ####
    ###########################
    def coordinate(self, xinit, hor, game_type="OLNE"):
        """
        Params:
            xinit [list] -- Joint initial state.
            hor [int] -- Time horizon to plan for (in number of steps)
            game_type [str] -- Type of games supported: 
                                OLNE: open-loop Nash, 
                                FBNE: feedback Nash, 
                                OLSB: open-loop Stackelberg
        Returns:
            xtraj [np.array] -- planned joint state traj. 
            uAtraj [np.array] -- planned control actions for agent A. 
            uBtraj [np.array] -- planned control actions for agent B. 
        """
        x0 = np.array(xinit)

        if game_type == "OLNE":
            # Note: for now, return a single solution (even if there are multiple). 
            xtraj, uAtraj, uBtraj = self.ol_nash_solve(x0, hor)
        elif game_type == "FBNE":
            xtraj, uAtraj, uBtraj = self.fb_nash_solve(x0, hor)
        elif game_type == "OLSB":
            xtraj, uAtraj, uBtraj = self.ol_stackelberg_solve(x0, hor)
        else:
            raise BaseException("undefined game type for {}".format(game_type))

        return xtraj, uAtraj, uBtraj

    def ol_nash_solve(self, x0, hor):
        raise NotImplementedError("Need to implement open-loop Nash solver!")

    def fb_nash_solve(self, x0, hor):
        raise NotImplementedError("Need to implement feedback Nash solver!")

    def ol_stackelberg_solve(self, x0, hor):
        raise NotImplementedError("Need to implement open-loop Stackelberg solver!")

    ###########################
    #### Utility functions ####
    ###########################

    def get_all_action_seq(self, hor):
        """
        Generates the set of all action sequences of 'hor' length. 
        Params: 
            hor [int] -- Time horizon to plan for (in number of steps) 
        Returns:
            [list of tuples] -- List of all action sequences. 
                                # of sequences: |U|^hor
        """
        action_list = list(range(self.actionsA))
        return list(itertools.product(action_list, repeat=hor))
    
    def get_legal_start(self):
        """
        Randomly generates a valid start state. 
        """
        poss_start = np.random.randint(0, self.S)
        while not self.is_legal_start(poss_start):
            poss_start = np.random.randint(0, self.S)
        return poss_start
    
    def is_legal_start(self, s):
        """
        Return if given state (in number form) is not within an obstacle or either goal.
        """
        return (not self.is_blocked(s)) and s != self.goalA and s != self.goalB
    
    def transition(self, s, aA, aB):
        """
        Given a *flattened* state, action of agent A and agent B,
        apply the transition function to get the next state.
        Params:
            s [int] -- Joint state.
            aA [int] -- Action agent A took.
            aB [int] -- Action agent B took.
        Returns:
            s_prime [int] -- Next state.
            illegal [bool] -- Whether the action taken was legal or not.
        """
        xA, yA, xB, yB = self.state_to_coor(s)
        x = [xA, yA, xB, yB]

        # call helper to do the heavy lifting. 
        x_prime, illegal = self.transition_helper(x, aA, aB)

        s_prime = self.coor_to_state(x_prime[0], x_prime[1], x_prime[2], x_prime[3] )
        if self.is_blocked(s_prime):
            illegal = True

        return s_prime, illegal

    def transition_helper(self, x, aA, aB):
        """
        Given a *unpacked* state, action of agent A and agent B,
        apply the transition function to get the next state.
        Params:
            x [int] --  joint state containing x-y coordinates for each agent
            aA [int] -- Action agent A took.
            aB [int] -- Action agent B took.
        Returns:
            xA_prime, yA_prime, xB_prime, yB_prime [int] -- Next state.
            illegal [bool] -- Whether the action taken was legal or not.
        """
        assert 0 <= aA < self.actionsA
        assert 0 <= aB < self.actionsB

        xA, yA, xB, yB = x[0], x[1], x[2], x[3]
        xA_prime, yA_prime, xB_prime, yB_prime = xA, yA, xB, yB
        if aA == SingleAgentActions.LEFT:
            xA_prime = xA - 1
        elif aA == SingleAgentActions.RIGHT:
            xA_prime = xA + 1
        elif aA == SingleAgentActions.DOWN:
            yA_prime = yA + 1
        elif aA == SingleAgentActions.UP:
            yA_prime = yA - 1
        else:
            raise BaseException("undefined action for agent A {}".format(aA))

        if aB == SingleAgentActions.LEFT:
            xB_prime = xB - 1
        elif aA == SingleAgentActions.RIGHT:
            xB_prime = xB + 1
        elif aA == SingleAgentActions.DOWN:
            yB_prime = yB + 1
        elif aA == SingleAgentActions.UP:
            yB_prime = yB - 1
        else:
            raise BaseException("undefined action for agent B {}".format(aB))

        illegal = False
        if xA_prime < 0 or xA_prime >= self.XA or yA_prime < 0 or yA_prime >= self.YA:
            illegal = True
            xA_prime = xA # "reset" the state if you went out of bounds. 
            yA_prime = yA
        elif xB_prime < 0 or xB_prime >= self.XB or yB_prime < 0 or yB_prime >= self.YB:
            illegal = True
            xB_prime = xB # "reset" the state if you went out of bounds. 
            yB_prime = yB

        return xA_prime, yA_prime, xB_prime, yB_prime, illegal

    def get_rewardA(self, x, aA, aB):
        """
        Calculate the reward for this joint state for agent A. 
        Params:
            x [list] -- joint state of agent A and B
            aA [int] -- action for agent A
            aB [int] -- action for agent B
        Return:
            reward [float] -- Agent A's instantaneous reward
        TODO: Can add more rewards here!
        """
        x_prime, _ = self.transition_helper(x, aA, aB)

        d_to_goalA = -np.linalg.norm(np.array([x_prime[0] - self.gA[0], x_prime[1] - self.gA[1]]), ord=2)**2
        d_coll = np.linalg.norm(np.array([x_prime[0] - x_prime[2], x_prime[1] - x_prime[3]]), ord=2)**2

        alpha = 1.0 # agent A's weight on collision cost. 

        return d_to_goalA + alpha * d_coll

    def get_rewardB(self, x, aA, aB):
        """
        Calculate the reward for this joint state for agent B. 
        Params:
            x [list] -- joint state of agent A and B
            aA [int] -- action for agent A
            aB [int] -- action for agent B
        Return:
            reward [float] -- Agent B's instantaneous reward
        TODO: Can add more rewards here!
        """
        x_prime, _ = self.transition_helper(x, aA, aB)

        d_to_goalB = -np.linalg.norm(np.array([x_prime[2] - self.gB[0], x_prime[3] - self.gB[1]]), ord=2)**2
        d_coll = np.linalg.norm(np.array([x_prime[0] - x_prime[2], x_prime[1] - x_prime[3]]), ord=2)**2
        
        beta = 1.0 # agent B's weight on collision cost. 

        return d_to_goalB + beta * d_coll
    
    def is_blocked(self, s):
        """
        Returns True if s is blocked.
        By default, state-action pairs that lead to blocked states are illegal.
        """
        if self.obstacles is None:
            return False

        # Check against internal representation of boxes. 
        xA, yA, xB, yB = self.state_to_coor(s)
        for box in self.obstacles:
            if xA >= box[0][0] and xA <= box[1][0] and yA >= box[1][1] and yA <= box[0][1]:
                return True
            if xB >= box[0][0] and xB <= box[1][0] and yB >= box[1][1] and yB <= box[0][1]:
                return True
        return False

    #################################
    # Conversion functions
    #################################
    # Helper functions convert between state number ("state") and discrete coordinates ("coor").
    #
    # State number ("state"):
    # A state `s` is an integer such that 0 <= s < self.S.
    #
    # Discrete coordinates ("coor"):
    # `x` is an integer such that 0 <= x < self.X. Increasing `x` corresponds to moving east.
    # `y` is an integer such that 0 <= y < self.Y. Increasing `y` corresponds to moving south.
    #################################

    def state_to_coor(self, s):
        """
        Params:
            s [int] -- The state.
        Returns:
            x, y -- The discrete coordinates corresponding to s.
        """
        #assert isinstance(s, int)
        assert 0 <= s < self.S
        yB = s % self.YA
        xB = ((s - yB) / self.YA) % self.XB
        yA = ((s - xB * self.YA - yB) / (self.YB * self.XB)) % self.YA
        xA = ((s - yA * self.YB * self.XB - xB * self.YA - yB) / (self.YB * self.XB * self.YA)) 
        return xA, yA, xB, yB

    def coor_to_state(self, xA, yA, xB, yB):
        """
        Convert discrete coordinates into a state, if that state exists.
        If no such state exists, raise a ValueError.
        The mapping is done in column-major order. 
        Params:
            xA, yA, xB, yB [int] -- The discrete x, y coordinates of each agent's state.
        Returns:
            s [int] -- The state.
        """

        xA, yA, xB, yB = int(xA), int(yA), int(xB), int(yB)
        if not(0 <= xA < self.XA):
            raise ValueError(xA, self.XA)
        if not (0 <= yA < self.YA):
            raise ValueError(yA, self.YA)
        if not(0 <= xB < self.XB):
            raise ValueError(xB, self.XB)
        if not (0 <= yB < self.YB):
            raise ValueError(yB, self.YB)

        return yB + self.YB * (xB + self.XB * (yA + self.YA * xA))

    def vis(self, xA, yA, xB, yB):
        """
        Visualizes the current joint state. 
        """
        # Create 2D world with obstacles on the map.
        world = 0.5*np.ones((self.YA, self.XA))

        # Add obstacles in the world in opaque color.
        if self.obstacles is not None:
            for obstacle in self.obstacles:
                lower = obstacle[0]
                upper = obstacle[1]
                world[upper[1]:lower[1]+1, lower[0]:upper[0]+1] = 1.0

        fig1, ax1 = plt.subplots()
        plt.imshow(world, cmap='Greys', interpolation='nearest')

        # Plot markers for start and goal
        st = self.state_to_coor(self.start)
        go = self.state_to_coor(self.goal)
        plt.scatter(xA, yA, c="red", marker="o", s=100)
        plt.scatter(xB, yB, c="blue", marker="o", s=100)
        plt.scatter(self.gA[0], self.gA[1], c="red", marker="x", s=300)
        plt.scatter(self.gB[0], self.gB[1], c="blue", marker="x", s=300)

        plt.xticks(range(self.XA), range(self.XA))
        plt.yticks(np.arange(0,self.YA),range(self.YA))
        # ax1.set_yticklabels([])
        # ax1.set_xticklabels([])
        # ax1.set_yticks()
        # ax1.set_xticks([])
        ax = plt.gca()
        # plt.minorticks_on
        ax.grid(True, which='both', color='black', linestyle='-', linewidth=1)
        plt.show()
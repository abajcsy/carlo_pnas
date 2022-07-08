import numpy as np
# We consider the cost: 1/2*x_t' Q1 x_t + 1/2*u_t'*R1u_t - q1'*x_t
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

x0 = np.array([-2,2,1.4,-1,2,2,-0.5,-4]).reshape(8,1) # initial condition
T = 90 # planning horizon







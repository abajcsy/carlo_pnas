import numpy as np

def OLNE_2player_solver(A,B1,B2,Q1,Q2,R1,R2,T,x0):
    T = T+1
    n = len(x0)
    m1 = B1.shape[1]
    m2 = B2.shape[1]
    M = np.zeros((T,2,n,n))
    M[T-1,0] = Q1
    M[T-1,1] = Q2
    Lambda = np.zeros((T-1,n,n))

    for iter in range(T-1):
        t = T-iter-1
        Lambda[t-1] = np.eye(n) + B1@np.linalg.inv(R1)@B1.T@M[t,0] + B2@np.linalg.inv(R2)@B2.T@M[t,1]
        M[t-1,0] = Q1 + A.T@M[t,0,:,:]@np.linalg.inv(Lambda[t-1,:,:])@A
        M[t-1,1] = Q2 + A.T@M[t,1,:,:]@np.linalg.inv(Lambda[t-1,:,:])@A

    x_traj = np.concatenate((x0.reshape(-1,1), np.zeros((n,T-1))),1)
    u1_traj = np.zeros((m1,T-1))
    u2_traj = np.zeros((m2,T-1))
    J1 = 1/2*x0.T@Q1@x0
    J2 = 1/2*x0.T@Q2@x0

    for t in range(T-1):
        u1 = -np.linalg.inv(R1) @ B1.T @ M[t+1,0,:,:] @ np.linalg.inv(Lambda[t,:,:]) @ A @ x_traj[:,t].reshape(n,1) 
        u1_traj[0,t] = u1[0]
        u1_traj[1,t] = u1[1]
        u2 = -np.linalg.inv(R2) @ B2.T @ M[t+1,1,:,:] @ np.linalg.inv(Lambda[t,:,:]) @ A @ x_traj[:,t].reshape(n,1)
        u2_traj[0,t] = u2[0]
        u2_traj[1,t] = u2[1]
        x_traj[:,t+1] = np.linalg.inv(Lambda[t]) @ A @ x_traj[:,t]
        J1 = J1 + 1/2*x_traj[:,t+1].T@Q1@x_traj[:,t+1] \
                + 1/2*u1_traj[:,t].T@R1@u1_traj[:,t]
        J2 = J2 + 1/2*x_traj[:,t+1].T@Q2@x_traj[:,t+1] \
                + 1/2*u2_traj[:,t].T@R2@u2_traj[:,t]

    return x_traj, u1_traj, u2_traj, J1, J2


def OLNE_quadratic_affine_cost_2player_solver(A,B1,B2,Q1,Q2,R1,R2,T,x0,q1,q2):
    # consider the cost: 1/2 x'Qx + q'x + 1/2 u'Ru
    # Sample codes: OLNE_quadratic_affine_cost_2player_solver(np.eye(2), np.ones((2,1)), np.ones((2,1)), np.eye(2), np.eye(2), ...
    # np.eye(1), np.eye(1), 4, np.ones((2,1)), np.ones((2,1)), np.ones((2,1)))
    
    n = len(x0)
    m1 = B1.shape[1]
    m2 = B2.shape[1]
    Q1_block = np.kron(np.eye(T), Q1)
    Q2_block = np.kron(np.eye(T), Q2)
    R1_block = np.kron(np.eye(T), R1)
    R2_block = np.kron(np.eye(T), R2)
    B1_block = np.kron(np.eye(T), B1)
    B2_block = np.kron(np.eye(T), B2)
    upper_triangle_matrix = np.zeros((T,T))
    upper_triangle_matrix[:T-1,1:] = np.eye(T-1)
    
    lower_triangle_matrix = np.zeros((T,T))
    lower_triangle_matrix[1:,:T-1] = np.eye(T-1)

    A_upper_block = np.kron(upper_triangle_matrix, -A.T) + np.eye(n*T)
    A_lower_block = np.kron(lower_triangle_matrix, -A) + np.eye(n*T)

    M = np.zeros(((3*n+m1+m2)*T, (3*n+m1+m2)*T))
    # The first layer block:
    M[:n*T, :n*T] = Q1_block
    M[:n*T, (n+m1+m2)*T:(2*n+m1+m2)*T] = A_upper_block
    # The second layer block:
    M[n*T:(n+m1)*T, n*T:(n+m1)*T] = R1_block
    M[n*T:(n+m1)*T, (n+m1+m2)*T:(2*n+m1+m2)*T] = -B1_block.T
    # The third layer block:
    M[(n+m1)*T:(2*n+m1)*T, :n*T] = Q2_block
    M[(n+m1)*T:(2*n+m1)*T, (2*n+m1+m2)*T:] = A_upper_block
    # The fourth layer block:
    M[(2*n+m1)*T:(2*n+m1+m2)*T, (n+m1)*T:(n+m1+m2)*T] = R2_block
    M[(2*n+m1)*T:(2*n+m1+m2)*T, (2*n+m1+m2)*T:] = - B2_block.T
    # The fifth layer block:
    M[(2*n+m1+m2)*T:, :n*T] = A_lower_block
    M[(2*n+m1+m2)*T:, n*T:(n+m1)*T] = -B1_block
    M[(2*n+m1+m2)*T:, (n+m1)*T:(n+m1+m2)*T] = -B2_block


    N = np.zeros(((3*n+m1+m2)*T, 1))
    N[:n*T] = np.kron(np.ones((T,1)), -q1)
    N[(n+m1)*T:(n+m1+n)*T] = np.kron(np.ones((T,1)), -q2)
    N[(2*n+m1+m2)*T:(2*n+m1+m2)*T+n] = A @ x0


    sol = np.linalg.inv(M) @ N

    x_traj = np.concatenate((x0.reshape(-1,1),  sol[:n*T].reshape(T,n).T ),1)
    u1_traj = sol[n*T:(n+m1)*T].reshape(T,m1).T
    u2_traj = sol[(n+m1)*T:(n+m1+m2)*T].reshape(T,m2).T
    J1 = 1/2*x0.T@Q1@x0 + q1.T@x0
    J2 = 1/2*x0.T@Q2@x0 + q2.T@x0

    for t in range(T):
        J1 = J1 + 1/2*x_traj[:,t+1].T@Q1@x_traj[:,t+1] \
                + 1/2*u1_traj[:,t].T@R1@u1_traj[:,t] \
                + q1.T @ x_traj[:,t+1]
        J2 = J2 + 1/2*x_traj[:,t+1].T@Q2@x_traj[:,t+1] \
                + 1/2*u2_traj[:,t].T@R2@u2_traj[:,t] \
                + q2.T @ x_traj[:,t+1]
    return x_traj, u1_traj, u2_traj, J1, J2

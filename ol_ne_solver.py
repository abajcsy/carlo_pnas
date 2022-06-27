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
import numpy as np

def FBNE_2player_solver(A,B1,B2,Q1,Q2,R1,R2,T,x0):
    T=T+1
    n = len(x0)
    m1 = B1.shape[1]
    m2 = B2.shape[1]
    K = np.zeros((T-1,2,m1,n))
    Z = np.zeros((T,2,n,n))
    F = np.zeros((T-1,n,n))
    Z[T-1,0] = Q1
    Z[T-1,1] = Q2
    for iter in range(T-1):
        t = T-iter-1
        big_matrix1 = np.concatenate((R1+B1.T@Z[t,0]@B1, B1.T@Z[t,0]@B2), 1)
        big_matrix2 = np.concatenate((B2.T@Z[t,1]@B1, R2+B2.T@Z[t,1]@B2), 1)
        big_matrix = np.concatenate((big_matrix1, big_matrix2),0)
        sol_K = np.linalg.inv(big_matrix)\
                @ np.concatenate((B1.T@Z[t,0]@A, B2.T@Z[t,1]@A),0)
        K[t-1,0] = sol_K[0:m1,:]
        K[t-1,1] = sol_K[m1:(m1+m2),:]
        F[t-1] = A - B1@K[t-1,0] - B2@K[t-1,1]
        Z[t-1,0] = F[t-1].T@Z[t,0]@F[t-1] + K[t-1,0].T@R1@K[t-1,0] + Q1
        Z[t-1,1] = F[t-1].T@Z[t,1]@F[t-1] + K[t-1,1].T@R2@K[t-1,1] + Q2
    x_traj = np.concatenate((x0.reshape(-1,1), np.zeros((n,T-1))),1)
    u1_traj = np.zeros((m1,T-1))
    u2_traj = np.zeros((m2,T-1))
    J1 = 1/2*x0.T@Q1@x0
    J2 = 1/2*x0.T@Q2@x0
    for t in range(T-1):
        u1_traj[:,t] = -K[t,0]@x_traj[:,t]
        u2_traj[:,t] = -K[t,1]@x_traj[:,t]
        x_traj[:,t+1] = F[t]@x_traj[:,t]
        J1 = J1 + 1/2*x_traj[:,t+1].T@Q1@x_traj[:,t+1] \
                + 1/2*u1_traj[:,t].T@R1@u1_traj[:,t]
        J2 = J2 + 1/2*x_traj[:,t+1].T@Q2@x_traj[:,t+1] \
                + 1/2*u2_traj[:,t].T@R2@u2_traj[:,t]
    return x_traj, u1_traj, u2_traj, J1, J2



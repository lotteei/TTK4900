
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------
# Code adapted from https://github.com/balzer82/Kalman
#-----------------------------------------------------------------------

def kalman(ID_xyz, save_rate):
    # Initial uncertainty
    P = 100.0 * np.eye(6)

    dt = 1/save_rate  # Time Step between Filter Steps 1/(fps = 15)

    # Transition matrix
    A = np.matrix([[1.0, 0.0, 0.0, dt, 0.0, 0.0],
                   [0.0, 1.0, 0.0, 0.0, dt, 0.0],
                   [0.0, 0.0, 1.0, 0.0, 0.0, dt],
                   [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 1.0], ])


    # Measurement matrix
    H = np.matrix([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])


    # Measurement noise covariance matrix R
    rp = 1.0 ** 2  # Noise of Position Measurement
    R = np.matrix([[rp, 0.0, 0.0],
                   [0.0, rp, 0.0],
                   [0.0, 0.0, rp]])


    # Process noise covariance matrix Q
    sj = 0.1
    Q = np.matrix([[(dt ** 6) / 36, 0, 0, (dt ** 5) / 12, 0, 0],
                   [0, (dt ** 6) / 36, 0, 0, (dt ** 5) / 12, 0],
                   [0, 0, (dt ** 6) / 36, 0, 0, (dt ** 5) / 12],
                   [(dt ** 5) / 12, 0, 0, (dt ** 4) / 4, 0, 0],
                   [0, (dt ** 5) / 12, 0, 0, (dt ** 4) / 4, 0],
                   [0, 0, (dt ** 5) / 12, 0, 0, (dt ** 4) / 4]]) * sj ** 2



    # Control matrix B
    B = np.matrix([[0.0],
                   [0.0],
                   [0.0],
                   [0.0],
                   [0.0],
                   [0.0]])

    u = 0.0


    I = np.eye(6)


    # Measurements
    X = []
    Y = []
    Z = []
    for xyz in ID_xyz:
        x = xyz[0]
        y = xyz[1]
        z = xyz[2]

        X.append(x)
        Y.append(y)
        Z.append(z)

   # add noise to the real 3D position
    m = int(len(X))

    sp = 0.1

    Xm = X + sp * (np.random.randn(m))
    Ym = Y + sp * (np.random.randn(m))
    Zm = Z + sp * (np.random.randn(m))

    measurements = np.vstack((Xm, Ym, Zm))


    # Initial states
    x = np.matrix([X[0], Y[0], Z[0], 0, 0, 0]).T


    ### Plotting
    xt = []
    yt = []
    zt = []
    dxt = []
    dyt = []
    dzt = []
    Zx = []
    Zy = []
    Zz = []
    Px = []
    Py = []
    Pz = []
    Pdx = []
    Pdy = []
    Pdz = []
    Kx = []
    Ky = []
    Kz = []
    Kdx = []
    Kdy = []
    Kdz = []

    ##### KALMAN FILTER ####################

    for filterstep in range(m):

        # Time Update (Prediction)
        x = A * x + B * u

        # Project the error covariance ahead
        P = A * P * A.T + Q

        # Measurement Update (Correction)
        S = H * P * H.T + R
        K = (P * H.T) * np.linalg.pinv(S)

        # Update the estimate via z
        Z = measurements[:, filterstep].reshape(H.shape[0], 1)
        y = Z - (H * x)
        x = x + (K * y)

        # Update the error covariance
        P = (I - (K * H)) * P

        # Save states for Plotting
        xt.append(float(x[0]))
        yt.append(float(x[1]))
        zt.append(float(x[2]))
        dxt.append(float(x[3]))
        dyt.append(float(x[4]))
        dzt.append(float(x[5]))

        Zx.append(float(Z[0]))
        Zy.append(float(Z[1]))
        Zz.append(float(Z[2]))
        Px.append(float(P[0, 0]))
        Py.append(float(P[1, 1]))
        Pz.append(float(P[2, 2]))
        Pdx.append(float(P[3, 3]))
        Pdy.append(float(P[4, 4]))
        Pdz.append(float(P[5, 5]))

        Kx.append(float(K[0, 0]))
        Ky.append(float(K[1, 0]))
        Kz.append(float(K[2, 0]))
        Kdx.append(float(K[3, 0]))
        Kdy.append(float(K[4, 0]))
        Kdz.append(float(K[5, 0]))
    
    '''
    ############## PLOT: Position in x/y plane ###########

    fig = plt.figure(figsize=(16, 9))

    plt.plot(xt, yt, label='Kalman filter', linewidth=5)
    plt.scatter(Xm, Ym, label='Measurement', c='gray', s=80)
    plt.legend(loc='best', fontsize=24)
    plt.axis('equal')
    plt.xlabel('x ($mm$)', fontsize=24)
    plt.ylabel('y ($mm$)', fontsize=24)
    plt.xticks(size=24)
    plt.yticks(size=24)
    plt.show()
    '''
    
    


    # Estimates for start position
    x_0 = xt[0]
    y_0 = yt[0]
    z_0 = zt[0]

    # Estimates for end position
    x_1 = xt[m - 1]
    y_1 = yt[m - 1]
    z_1 = zt[m - 1]

    # Estimates for start velocity
    dx_0 = dxt[0]
    dy_0 = dyt[0]
    dz_0 = dzt[0]

    # Estimates for end velocity
    dx_1 = dxt[m - 1]
    dy_1 = dyt[m - 1]
    dz_1 = dzt[m - 1]

    return x_0, y_0, z_0, x_1, y_1, z_1, dx_0, dy_0, dz_0, dx_1, dy_1, dz_1












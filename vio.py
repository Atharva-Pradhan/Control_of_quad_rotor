#%% Imports

import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm
from scipy.spatial.transform import Rotation


#%% Functions

def nominal_state_update(nominal_state, w_m, a_m, dt):
    """
    function to perform the nominal state update

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                    all elements are 3x1 vectors except for q which is a Rotation object
    :param w_m: 3x1 vector - measured angular velocity in radians per second
    :param a_m: 3x1 vector - measured linear acceleration in meters per second squared
    :param dt: duration of time interval since last update in seconds
    :return: new tuple containing the updated state
    """
    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state
    # YOUR CODE HERE
    
    new_p = p + v*dt + 0.5*(q.as_matrix()@(a_m - a_b)+g)*(dt**2)
    new_v = v + (q.as_matrix()@(a_m - a_b)+g)*dt
    update_q = (w_m - w_b)*dt
    update_q = np.reshape(update_q,(3,))
    update_q = Rotation.from_rotvec(update_q)
    new_q = quat_prod(q.as_quat(),update_q.as_quat())
    new_q = Rotation.from_quat(new_q)

    return new_p, new_v, new_q, a_b, w_b, g


def error_covariance_update(nominal_state, error_state_covariance, w_m, a_m, dt,
                            accelerometer_noise_density, gyroscope_noise_density,
                            accelerometer_random_walk, gyroscope_random_walk):
    """
    Function to update the error state covariance matrix

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                        all elements are 3x1 vectors except for q which is a Rotation object
    :param error_state_covariance: 18x18 initial error state covariance matrix
    :param w_m: 3x1 vector - measured angular velocity in radians per second
    :param a_m: 3x1 vector - measured linear acceleration in meters per second squared
    :param dt: duration of time interval since last update in seconds
    :param accelerometer_noise_density: standard deviation of accelerometer noise
    :param gyroscope_noise_density: standard deviation of gyro noise
    :param accelerometer_random_walk: accelerometer random walk rate
    :param gyroscope_random_walk: gyro random walk rate
    :return:
    """

    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state

    # YOUR CODE HERE
    v_i = (accelerometer_noise_density**2)*(dt**2)*np.eye(3)
    theta_i = (gyroscope_noise_density**2)*(dt**2)*np.eye(3)
    A_i = (accelerometer_random_walk**2)*(dt)*np.eye(3)
    omega_i = (gyroscope_random_walk**2)*(dt)*np.eye(3)
    Q_i = np.zeros((12,12))
    Q_i[0:3,0:3] = v_i
    Q_i[3:6,3:6] = theta_i
    Q_i[6:9,6:9] = A_i
    Q_i[9:12,9:12] = omega_i

    F_i = np.zeros((18,12))
    F_i[3:6,0:3] = np.eye(3)
    F_i[6:9,3:6] = np.eye(3)
    F_i[9:12,6:9] = np.eye(3)
    F_i[12:15,9:12] = np.eye(3)


    F_x = np.zeros((18,18))

    # 1st Row
    F_x[0:3,0:3] = np.eye(3)
    F_x[0:3,3:6] = np.eye(3)*dt

    # 2nd Row
    F_x[3:6,3:6] = np.eye(3)
    w = a_m-a_b
    w = np.reshape(w,(3,))
    w = np.array([[0,-w[2],w[1]],
              [w[2],0,-w[0]],
              [-w[1],w[0],0]])
    s = -(q.as_matrix()@w)*dt
    F_x[3:6,6:9] = s
    F_x[3:6,9:12] = -q.as_matrix()*dt
    F_x[3:6,15:18] = np.eye(3)*dt

    # 3rd Row
    update_q = (w_m - w_b)*dt
    update_q = np.reshape(update_q,(3,))
    update_q = Rotation.from_rotvec(update_q)
    F_x[6:9,6:9] = (update_q.as_matrix()).T
    F_x[6:9,12:15] = -np.eye(3)*dt

    # 4th Row
    F_x[9:12,9:12] = np.eye(3)

    # 5th Row
    F_x[12:15,12:15] = np.eye(3)

    # 6th Row
    F_x[15:18,15:18] = np.eye(3)

    P = F_x@error_state_covariance@(F_x.T) + F_i@Q_i@(F_i.T)

    # return an 18x18 covariance matrix
    return P


def measurement_update_step(nominal_state, error_state_covariance, uv, Pw, error_threshold, Q):
    """
    Function to update the nominal state and the error state covariance matrix based on a single
    observed image measurement uv, which is a projection of Pw.

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                        all elements are 3x1 vectors except for q which is a Rotation object
    :param error_state_covariance: 18x18 initial error state covariance matrix
    :param uv: 2x1 vector of image measurements
    :param Pw: 3x1 vector world coordinate
    :param error_threshold: inlier threshold
    :param Q: 2x2 image covariance matrix
    :return: new_state_tuple, new error state covariance matrix
    """
    
    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state
    P_c = (q.as_matrix().T)@(Pw - p)
    A = np.array([P_c[0,0]/P_c[2,0],P_c[1,0]/P_c[2,0]])
    A = np.reshape(A,(2,1))
    innovation = uv - A
    if (np.linalg.norm(innovation)<error_threshold):
        dztdpc = np.array([[1,0,-A[0,0]],[0,1,-A[1,0]]])*(1/P_c[2,0])
        dpcdtheta = np.array([[0,-P_c[2,0],P_c[1,0]],
                              [P_c[2,0],0,-P_c[0,0]],
                              [-P_c[1,0],P_c[0,0],0]])
        dpcddp = -q.as_matrix().T
        dztdtheta = dztdpc@dpcdtheta
        dztddp = dztdpc@dpcddp
        Ht = np.zeros((2,18))
        Ht[0:2,0:3] = dztddp
        Ht[0:2,6:9] = dztdtheta
        Kt = (error_state_covariance@(Ht.T))@np.linalg.inv(Ht@error_state_covariance@(Ht.T)+Q)
        error_state_covariance = (np.eye(18)-Kt@Ht)@error_state_covariance@((np.eye(18)-Kt@Ht).T) + Kt@Q@(Kt.T)
        delx = Kt@innovation
        delp = delx[0:3,0]
        delp = np.reshape(delp,(3,1))
        delv = delx[3:6,0]
        delv = np.reshape(delv,(3,1))
        delq = delx[6:9,0]
        delab = delx[9:12,0]
        delab = np.reshape(delab,(3,1))
        delwb = delx[12:15,0]
        delwb = np.reshape(delwb,(3,1))
        delg = delx[15:18,0]
        delg = np.reshape(delg,(3,1))
        p = p+delp
        v = v+delv
        q = quat_prod(q.as_quat(),Rotation.from_rotvec(delq).as_quat())
        q = Rotation.from_quat(q)
        a_b = a_b+delab
        w_b = w_b+delwb
        g = g+delg
    # YOUR CODE HERE - compute the innovation next state, next error_state covariance
    # innovation = np.zeros((2, 1))
    

    return (p, v, q, a_b, w_b, g), error_state_covariance, innovation


def quat_prod(u,v):
    """
    Input:
        u - First Quaternion
        v - Second Quaternion
    Output
        q_quat - uv (quaternion multiplication) 
    """
    u = np.reshape(u,(4,))
    v = np.reshape(v,(4,))
    q_const = np.array([u[3]*v[3]-u[0:3]@v[0:3]])
    q_axis  = np.array([u[3]*v[0:3]+v[3]*u[0:3] + np.cross(u[0:3],v[0:3])])
    q_quat  = np.append(q_axis,q_const)

    return q_quat
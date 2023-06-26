import numpy as np
from scipy.spatial.transform import Rotation as R

class SE3Control(object):
    """

    """
    def __init__(self, quad_params):
        """
        This is the constructor for the SE3Control object. You may instead
        initialize any parameters, control gain values, or private state here.

        For grading purposes the controller is always initialized with one input
        argument: the quadrotor's physical parameters. If you add any additional
        input arguments for testing purposes, you must provide good default
        values!

        Parameters:
            quad_params, dict with keys specified by crazyflie_params.py

        """

        # Quadrotor physical parameters.
        self.mass            = quad_params['mass'] # kg
        self.Ixx             = quad_params['Ixx']  # kg*m^2
        self.Iyy             = quad_params['Iyy']  # kg*m^2
        self.Izz             = quad_params['Izz']  # kg*m^2
        self.arm_length      = quad_params['arm_length'] # meters
        self.rotor_speed_min = quad_params['rotor_speed_min'] # rad/s
        self.rotor_speed_max = quad_params['rotor_speed_max'] # rad/s
        self.k_thrust        = quad_params['k_thrust'] # N/(rad/s)**2
        self.k_drag          = quad_params['k_drag']   # Nm/(rad/s)**2

        # You may define any additional constants you like including control gains.
        self.inertia = np.diag(np.array([self.Ixx, self.Iyy, self.Izz])) # kg*m^2
        self.g = 9.81 # m/s^2

        # STUDENT CODE HERE
        # self.kp = np.diag(np.array([25,5,20]))
        # self.kd = np.diag(np.array([4,6.5,5.65]))
        # self.kr = np.diag(np.array([1800,2500,2000]))
        # self.kw = np.diag(np.array([80,80,1000]))

        # self.kp = np.diag(np.array([10,10,50]))
        # self.kd = np.diag(np.array([5.2,6.5,5.5]))
        # self.kr = np.diag(np.array([2600,2600,150]))
        # self.kw = np.diag(np.array([130,130,80]))
        
        # self.kp = np.diag(np.array([25, 25, 50]))
        # self.kd = np.diag(np.array([50, 50, 20]))
        # self.kr = np.diag(np.array([1600, 1600, 20]))
        # self.kw = np.diag(np.array([60, 60, 15]))
        
        # self.kp = np.diag(np.array([25, 20, 40]))
        # self.kd = np.diag(np.array([5, 10, 20]))
        # self.kr = np.diag(np.array([1600, 1600, 300]))
        # self.kw = np.diag(np.array([10, 10, 100]))
        
        # self.kp = np.diag(np.array([15, 15, 40]))         # ROHIT
        # self.kd = np.diag(np.array([4.4, 50, 7]))
        # self.kr = np.diag(np.array([2600, 2600, 150]))
        # self.kw = np.diag(np.array([130, 130, 80]))

        # self.kp = np.diag(np.array([11, 12.5, 40]))            # Worked Velo 3 perfect
        # self.kd = np.diag(np.array([5, 5, 6.5]))
        # self.kr = np.diag(np.array([2600, 2600, 150]))
        # self.kw = np.diag(np.array([130, 130, 80]))
        
        self.kp = np.diag(np.array([7.5, 7.5, 22]))
        self.kd = np.diag(np.array([4.4, 4.2, 6.5]))
        self.kr = np.diag(np.array([2600, 2600, 150]))
        self.kw = np.diag(np.array([130, 130, 80]))

        # self.kp = np.diag(np.array([1, 10, 20]))
        # self.kd = np.diag(np.array([7, 7, 5]))
        # self.kr = np.diag(np.array([2600, 2600, 150]))
        # self.kw = np.diag(np.array([130, 130, 80]))

        
        # self.kf = 6.11e-8
        # self.km = 1.5e-9

    def update(self, t, state, flat_output):
        """
        This function receives the current time, true state, and desired flat
        outputs. It returns the command inputs.

        Inputs:
            t, present time in seconds
            state, a dict describing the present state with keys
                x, position, m
                v, linear velocity, m/s
                q, quaternion [i,j,k,w]
                w, angular velocity, rad/s
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s

        Outputs:
            control_input, a dict describing the present computed control inputs with keys
                cmd_motor_speeds, rad/s
                cmd_thrust, N (for debugging and laboratory; not used by simulator)
                cmd_moment, N*m (for debugging; not used by simulator)
                cmd_q, quaternion [i,j,k,w] (for laboratory; not used by simulator)
        """

        ep = flat_output["x"] - state["x"]
        ev = (flat_output["x_dot"] - state["v"])
        q = state["q"]
        r_double_dot_des = flat_output["x_ddot"] + np.matmul(self.kp,ep) + np.matmul(self.kd,ev)
        F_des = self.mass*r_double_dot_des + np.array([0,0,self.mass*self.g])
        R_B_to_A = R.from_quat(state["q"])
        R_B_to_A = R_B_to_A.as_matrix()
        u1 = np.matmul(np.matmul(R_B_to_A,np.array([0,0,1])),F_des)

        b3_des = F_des/np.linalg.norm(F_des)
        a_psi = np.array([np.cos(flat_output["yaw"]), np.sin(flat_output["yaw"]), 0])
        b2_des = np.cross(b3_des,a_psi)/np.linalg.norm(np.cross(b3_des,a_psi))
        b1_des = np.cross(b2_des,b3_des)

        R_des = np.c_[b1_des,b2_des,b3_des]
        A = np.transpose(R_des)@R_B_to_A - np.transpose(R_B_to_A)@R_des
        A = np.array([A[2,1], A[0,2], A[1,0]])
        er = 0.5*A
        ew = state["w"]
        B = -(np.matmul(self.kr,er) + np.matmul(self.kw,ew))
        u2 = np.matmul(self.inertia, B)

        gamma = self.k_drag/self.k_thrust
        U = np.append(u2,u1)
        A = np.array([[0, self.arm_length, 0, -self.arm_length],
                      [-self.arm_length, 0, self.arm_length, 0],
                      [gamma, -gamma, gamma, -gamma],[1,1,1,1]
                      ])
        F = np.matmul(np.linalg.inv(A),U)
        cmd_motor_speeds = np.sqrt(np.absolute(F/self.k_thrust))
        cmd_motor_speeds = np.sign(F)*cmd_motor_speeds
        cmd_moment = u2
        cmd_thrust = u1
        cmd_motor_speeds = np.clip(cmd_motor_speeds,self.rotor_speed_min,self.rotor_speed_max)
        cmd_q = q

        # STUDENT CODE HERE

        control_input = {'cmd_motor_speeds':cmd_motor_speeds,
                         'cmd_thrust':cmd_thrust,
                         'cmd_moment':cmd_moment,
                         'cmd_q':cmd_q}
        return control_input


if __name__ == "__main__":
    x        = np.ones((3,))
    x_dot    = np.ones((3,))
    x_ddot   = np.zeros((3,))
    x_dddot  = np.zeros((3,))
    x_ddddot = np.zeros((3,))
    yaw = 0
    yaw_dot = 0
    xa = np.zeros((3,))
    v = np.zeros((3,))
    q = np.array([0,0,0,1])
    w = np.zeros((3,))
    t = 0
    flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot}
    state = {'x':xa,'v':v,'q':q,'w':w}
    quad_params = {
    'mass': 0.03,   # kg
    'Ixx':  1.43e-5, # kg*m^2
    'Iyy':  1.43e-5, # kg*m^2
    'Izz':  2.89e-5, # kg*m^2
    'arm_length': 0.046, # meters
    'rotor_speed_min': 0,    # rad/s
    'rotor_speed_max': 2500, # rad/s
    'k_thrust': 2.3e-08, # N/(rad/s)**2
    'k_drag':   7.8e-11, # Nm/(rad/s)**2
}
    se = SE3Control(quad_params)
    cp = se.update(t,state,flat_output)
    # print("cp", cp)
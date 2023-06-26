import numpy as np
import math
from scipy.sparse.linalg import spsolve as sp
from scipy.sparse import lil_matrix as lm

# from graph_search import graph_search
from .graph_search import graph_search

class WorldTraj(object):
    """
    """
    def perpendicularDistance(self, point, line):
        x, y, z = point
        x1, y1, z1 = line[0]
        x2, y2, z2 = line[1]

        if x1 == x2 and y1 == y2 and z1 == z2:
            return math.sqrt((x - x1)**2 + (y - y1)**2 + (z - z1)**2)

        denominator = (x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2

        if denominator == 0:
            return math.sqrt((x - x1)**2 + (y - y1)**2 + (z - z1)**2)

        t = ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1) + (z - z1) * (z2 - z1)) / denominator
        t = max(0, min(1, t))

        x_proj = x1 + t * (x2 - x1)
        y_proj = y1 + t * (y2 - y1)
        z_proj = z1 + t * (z2 - z1)

        return math.sqrt((x - x_proj)**2 + (y - y_proj)**2 + (z - z_proj)**2)

    def Line(self, p1, p2):
        return (p1, p2)
    
    def DouglasPeucker(self, PointList, epsilon):
        dmax = 0
        index = 0
        end = len(PointList)
        for i in range(1, end - 1):
            d = self.perpendicularDistance(PointList[i], self.Line(PointList[0], PointList[end - 1]))
            if d > dmax:
                index = i
                dmax = d
        ResultList = []

        if dmax > epsilon:
            recResults1 = self.DouglasPeucker(PointList[:index + 1], epsilon)
            recResults2 = self.DouglasPeucker(PointList[index:], epsilon)
            ResultList = recResults1[:-1] + recResults2
        else:
            ResultList = [PointList[0], PointList[end - 1]]
        return ResultList
    
    def prune(self, path):
        pt_list = list(self.path)
        iter = 0
        while iter != len(pt_list) - 2:
            if iter > len(pt_list) - 2:
                break
            this = pt_list[iter]-pt_list[iter+1]
            next = pt_list[iter+1]-pt_list[iter+2]
            direction = np.cross(this, next)   
            distance = np.linalg.norm(this)     
            norm_direction = np.linalg.norm(direction)
            if norm_direction == 0:
                del pt_list[iter + 1]
                iter =iter- 1
            elif distance > 0.01:
                del pt_list[iter]
            iter = iter + 1
        points = np.array(pt_list)
        return points
    
    def X(self,t0):
        t0=t0[0]
        X = np.array([[0,            0,           0,          0,          0,         0,      0,  1],
                      [t0**7,        t0**6,       t0**5,      t0**4,      t0**3,     t0**2,  t0, 1],
                      [7*(t0**6),    6*(t0**5),   5*(t0**4),  4*(t0**3),  3*(t0**2), 2*(t0), 1,  0],
                      [42*(t0**5),   30*(t0**4),  20*(t0**3), 12*(t0**2), 6*(t0),    2,      0,  0],
                      [210*(t0**4),  120*(t0**3), 60*(t0**2), 24*(t0),    6,         0,      0,  0],
                      [840*(t0**3),  360*(t0**2), 120*(t0),   24,         0,         0,      0,  0],
                      [2520*(t0**2), 720*(t0),    120,        0,          0,         0,      0,  0],
                      [5040*(t0),    720,         0,          0,          0,         0,      0,  0]])
        return X
    
    def get_final_A(self, matrix2, count, time, ar):
        block = [-1, -2, -6, -24, -120, -720]
        idxs = [5, 6, 7, 8, 9, 10]
        if count != len(time) - 1:
            for i, val in enumerate(block):
                matrix2[idxs[i] + 8*count, 14 - i + 8*count] = val
            matrix2[8*count + 3:8*count + 11, 8*count:8*count + 8] = ar
        else:
            matrix2[8*count + 3:8*count + 11, 8*count:8*count + 8] = ar[:5, :]
        return matrix2
        
    def get_mat_A(self,m,t_start):
        A = lm((8*m,8*m))
        total_time = np.sum(t_start)
        segment = 0
        for t0 in t_start:
            var_mat = self.X(t0)
            A[[0, 1, 2], [6, 5, 4]] = [1, 2, 6]
            A0 = self.get_final_A(A, segment, t_start, var_mat)
            segment+=1
        return total_time, A0
    
    def get_mat_B(self,m):
        B = lm((8*m,3))
        for i in range(m):
            B[8*i + 3] = self.points[i]
            B[8*i + 4] = self.points[i + 1]
        return B
    
    def get_states(self,total_time,t,t_start_i,C):
        for i in range(np.shape(self.points)[0]-1):
            if t<t_start_i[i+1] and t>=t_start_i[i]:
                t0 = t - t_start_i[i]
                D = np.array([[t0**7,       t0**6,       t0**5,      t0**4,      t0**3,     t0**2,  t0, 1],
                                [7*(t0**6),   6*(t0**5),   5*(t0**4),  4*(t0**3),  3*(t0**2), 2*(t0), 1,  0],
                                [42*(t0**5),  30*(t0**4),  20*(t0**3), 12*(t0**2), 6*(t0),    2,      0,  0],
                                [210*(t0**4), 120*(t0**3), 60*(t0**2), 24*(t0),    6,         0,      0,  0],
                                [840*(t0**3), 360*(t0**2), 120*(t0),   24,         0,         0,      0,  0]])
                    
                E = D@C[8*i:8*i+8,:]
                x = E[0,:]
                x_dot = E[1,:]
                x_ddot = E[2,:]
                x_dddot = E[3,:]
                x_ddddot = E[4,:]
                break
            elif t>=t_start_i[-1]:
                x = self.points[-1]
                x_dot = np.zeros((3,))
                x_ddot = np.zeros((3,))
                x_dddot = np.zeros((3,))
                x_ddddot = np.zeros((3,))
                break
        return x, x_dot, x_ddot, x_dddot, x_ddddot

    def __init__(self, world, start, goal):
    
        """
        This is the constructor for the trajectory object. A fresh trajectory
        object will be constructed before each mission. For a world trajectory,
        the input arguments are start and end positions and a world object. You
        are free to choose the path taken in any way you like.

        You should initialize parameters and pre-compute values such as
        polynomial coefficients here.

        Parameters:
            world, World object representing the environment obstacles
            start, xyz position in meters, shape=(3,)
            goal,  xyz position in meters, shape=(3,)

        """

        # You must choose resolution and margin parameters to use for path
        # planning. In the previous project these were provided to you; now you
        # must chose them for yourself. Your may try these default values, but
        # you should experiment with them!
        self.resolution = np.array([0.25, 0.25, 0.25])
        self.margin = 0.5
        print(start)
        if start[2] == 0.7:
            self.velocity = 11.5
        elif start[2]==1.5:
            self.velocity = 8.3
        else:
            self.velocity = 8.5
        self.start = start
        # You must store the dense path returned from your Dijkstra or AStar
        # graph search algorithm as an object member. You will need it for
        # debugging, it will be used when plotting results.
        self.path, _ = graph_search(world, self.resolution, self.margin, start, goal, astar=True)

        # You must generate a sparse set of waypoints to fly between. Your
        # original Dijkstra or AStar path probably has too many points that are
        # too close together. Store these waypoints as a class member; you will
        # need it for debugging and it will be used when plotting results.
        
        # print(np.shape(self.path))
        # Finally, you must compute a trajectory through the waypoints similar
        # to your task in the first project. One possibility is to use the
        # WaypointTraj object you already wrote in the first project. However,
        # you probably need to improve it using techniques we have learned this
        # semester.
        # STUDENT CODE HERE

        # pt_list = list(self.path)
        # iter = 0
        # while iter != len(pt_list) - 2:
        #     if iter > len(pt_list) - 2:
        #         break
        #     this = pt_list[iter]-pt_list[iter+1]
        #     next = pt_list[iter+1]-pt_list[iter+2]
        #     direction = np.cross(this, next)   
        #     distance = np.linalg.norm(this)     
        #     norm_direction = np.linalg.norm(direction)
        #     if norm_direction == 0:
        #         del pt_list[iter + 1]
        #         iter =iter- 1
        #     elif distance > 0.01:
        #         del pt_list[iter]
        #     iter = iter + 1
        # self.points = np.array(pt_list)
        # self.points = self.path
        # self.points = self.DouglasPeucker(self.path,0.005)
        # self.points = np.array(self.points)
        self.points = self.prune(self.path)
        self.distance=np.linalg.norm(self.points[1::]-self.points[0:-1], axis=1).reshape(-1, 1)

    def update(self, t):
        """
        Given the present time, return the desired flat output and derivatives.

        Inputs
            t, time, s
        Outputs
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s
        """
        x        = np.zeros((3,))
        x_dot    = np.zeros((3,))
        x_ddot   = np.zeros((3,))
        x_dddot  = np.zeros((3,))
        x_ddddot = np.zeros((3,))
        yaw = 0
        yaw_dot = 0
        # print(self.start)
        # STUDENT CODE HERE
        t_start = self.distance/self.velocity
        t_start[0]*=3
        t_start[-1]*=3
        t_start = t_start*np.sqrt(1.65/t_start)
        t_start_i = np.vstack((np.zeros(1), np.cumsum(t_start, axis=0))).flatten()
        t_start = t_start.clip(0.25,np.inf)
        
        m = np.shape(self.points)[0] - 1
        
        total_time, A = self.get_mat_A(m,t_start)
        B = self.get_mat_B(m)

        A = A.tocsc()
        C = sp(A, B).toarray()
        
        x, x_dot, x_ddot, x_dddot, x_ddddot = self.get_states(total_time,t,t_start_i,C)
        yaw = 0
        yaw_dot = 0

        flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot}
        return flat_output
    
if __name__ == "__main__":
    from flightsim.world import World
    from pathlib import Path
    import inspect
    # points = np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1,1,1]])
    # Choose a test example file. You should write your own example files too!
    # filename = '../util/test_over_under.json'
    filename = '../util/test_window.json'
    # filename = '../util/test_maze.json'

    # Load the test example.
    file = Path(inspect.getsourcefile(lambda:0)).parent.resolve() / '..' / 'util' / filename
    world = World.from_file(file)          # World boundary and obstacles.
    start  = world.world['start']          # Start point, shape=(3,)
    goal   = world.world['goal']           # Goal point, shape=(3,)

    start = np.array(
    [0, 0, 0])
    goal = np.array([1,1,1])
    # points = np.array([
    #  [0.0, 0.0, 0.0],
    #     [2.0, 0.0, 0.0],
    #     [2.0, 2.0, 0.0],
    #     [2.0, 2.0, 2.0],
    #     [0.0, 2.0, 2.0],
    #     [0.0, 0.0, 2.0]])
    # points = np.array([[0,0,0],[1,0,0]])
    # print("S",start,"G",goal)
    wp = WorldTraj(world, start, goal)
    for t in range(3):
        flat_output = wp.update(t)
        print("flat_output",flat_output)
from heapq import heappush, heappop, heapify  # Recommended.
import numpy as np

from flightsim.world import World
from collections import defaultdict
# from occupancy_map import OccupancyMap # Recommended.
from .occupancy_map import OccupancyMap

def neighbors(occ_map, voxel):
    """
    Parameters:
        voxel,      The voxel index whose neighbors are needed
    Output
        neighbors,  Possible neighbors of that voxel
    """
    m = [-1, 0, 1]
    motion = np.stack(np.meshgrid(m, m, m), axis=-1).reshape(-1, 3)
    motion = np.delete(motion,13,axis=0)
    neighbor = motion + voxel
    # possible_or_not = []
    # i = 0
    # for a in neighbor:        
    #     possible_or_not = np.append(possible_or_not,bool(occ_map.is_valid_index(a) and not occ_map.is_occupied_index(a)))
    # main = neighbor[possible_or_not==1]
    return neighbor
    
def cost(u,v,occ_map):
    """
    Parameters:
        u,          
        v,          
    Output:
        dist,       
    """
    u = occ_map.index_to_metric_center(u)
    v = occ_map.index_to_metric_center(v)
    dist = np.linalg.norm(u-v)

    return dist

def graph_search(world, resolution, margin, start, goal, astar):
    """
    Parameters:
        world,      World object representing the environment obstacles
        resolution, xyz resolution in meters for an occupancy map, shape=(3,)
        margin,     minimum allowed distance in meters from path to obstacles.
        start,      xyz position in meters, shape=(3,)
        goal,       xyz position in meters, shape=(3,)
        astar,      if True use A*, else use Dijkstra
    Output:
        return a tuple (path, nodes_expanded)
        path,       xyz position coordinates along the path in meters with
                    shape=(N,3). These are typically the centers of visited
                    voxels of an occupancy map. The first point must be the
                    start and the last point must be the goal. If no path
                    exists, return None.
        nodes_expanded, the number of nodes that have been expanded
    """

    # While not required, we have provided an occupancy map you may use or modify.
    path_final = None
    occ_map = OccupancyMap(world, resolution, margin)
    # print(occ_map)
    # Retrieve the index in the occupancy grid matrix corresponding to a position in space.
    start_index = tuple(occ_map.metric_to_index(start))
    goal_index = tuple(occ_map.metric_to_index(goal))
    final = start_index

    Q = [(0,start_index)]
    Y = []
    p = {start_index: None}
    u = start_index
    dist = defaultdict(lambda: np.inf)
    dist[start_index] = 0
    
    g = defaultdict(lambda: np.inf)
    g[start_index] = 0
    if astar:
        nodes_expanded = 0
        while Q:
            nodes_expanded += 1
            (current_val,current) = heappop(Q)

            if(current==goal_index):
                final=[]
                final.append(goal)
                m = p[goal_index]

                while m:
                    final.append(occ_map.index_to_metric_center(m))
                    m = p[m]

                final.append(start)
                path_final=np.flip(final, axis=0)
                break

            u = tuple(current)
            # heappush(Y,tuple(current))
            a = neighbors(occ_map,current)
            for neigh in a:
                if occ_map.is_valid_index(neigh) and not (occ_map.is_occupied_index(u)):
                    d = current_val + cost(u,neigh,occ_map) + cost(u,goal_index,occ_map)            # Hueristic
                    if (d<dist[tuple(neigh)]) or (tuple(neigh) not in dist):
                        dist[tuple(neigh)] = d
                        p[tuple(neigh)] = u
                        heappush(Q,(dist[tuple(neigh)],tuple(neigh)))
        # Return a tuple (path, nodes_expanded)
        return path_final, nodes_expanded
    

    else:
        nodes_expanded = 0
        while Q:
            nodes_expanded += 1
            (current_val,current) = heappop(Q)

            if(current==goal_index):
                final=[]
                final.append(goal)
                m = p[goal_index]

                while m:
                    final.append(occ_map.index_to_metric_center(m))
                    m = p[m]

                final.append(start)
                path_final=np.flip(final, axis=0)
                break

            u = tuple(current)
            # heappush(Y,tuple(current))
            a = neighbors(occ_map,current)
            for neigh in a:
                if occ_map.is_valid_index(neigh) and not (occ_map.is_occupied_index(u)):
                    d = current_val + cost(u,neigh,occ_map)
                    if (d<dist[tuple(neigh)]) or (tuple(neigh) not in dist):
                        dist[tuple(neigh)] = d
                        p[tuple(neigh)] = u
                        heappush(Q,(dist[tuple(neigh)],tuple(neigh)))
        # Return a tuple (path, nodes_expanded)
        return path_final, nodes_expanded
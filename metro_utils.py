from __future__ import annotations
import numpy as np
from itertools import permutations, chain, combinations  
from math import ceil
from station import Station
from city import City
import matplotlib.pyplot as plt
from numpy.random import shuffle, randint, choice
from copy import deepcopy
from time import time

SMALL, MED, LARGE, LW = 18, 24, 30, 3
plt.rc('axes', titlesize=MED)    # fontsize of the axes title
plt.rc('axes', labelsize=MED)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL) # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL) # fontsize of the tick labels
plt.rc('legend', fontsize=MED)   # legend fontsize
plt.rc('font', size=LARGE)       # controls default text sizes
HORIZONTAL, VERTICAL = 0, 1
    
def midpoint(s1: Station, s2: Station) -> tuple[float, float]:
    ''' Find the midpoint - the point where the bend is on the track between stations '''
    minn, maxx = get_extreme_abs_difference(s1, s2)

    straight_len = maxx - minn

    Δx = s1.x - s2.x
    Δy = s1.y - s2.y

    alignment = HORIZONTAL if abs(Δx) >= abs(Δy) else VERTICAL
    dirx, diry = np.sign(Δx), np.sign(Δy)

    midx = s1.x - dirx * (alignment == HORIZONTAL) * straight_len
    midy = s1.y - diry * (alignment == VERTICAL  ) * straight_len

    return midx, midy

def gen_distance_matrix(city: City) -> np.ndarray:
    ''' Creates nxn array of distances between stations '''
    dist_matrix = np.zeros((city.n_stations, city.n_stations))
    
    for s1 in city.stations:
        for s2 in city.stations:
            dist_matrix[s1.id, s2.id] = distance(s1, s2)
            
    return dist_matrix

def distance(s1: Station, s2: Station) -> float:
    ''' Returns distance of rail between two adjacent stations '''
    minn, maxx = get_extreme_abs_difference(s1, s2)
    return maxx + (2**0.5 - 1) * minn

def get_extreme_abs_difference(s1: Station, s2: Station) -> tuple[float, float]:
    lΔxl = abs(s1.x - s2.x)
    lΔyl = abs(s1.y - s2.y)

    return min(lΔxl, lΔyl), max(lΔxl, lΔyl)

def total_distance(rails):
    ''' For a given track allocation, returns total distance
    rails can be a list of station pairs [(1, 2), (2, 3), (3, 1)]
    or a list of lists of stations pairs [[(0, 1), (1, 0)], [(2, 3), (3, 4), (4, 3)]]
    '''
    # Check if rails is just one rail
    if isinstance(rails[0], Station):
        rails = [rails]
        
    dist = 0
    for rail in rails:
        start = rail[0]
        for end in rail[1:]:
            dist += distance(start, end)
            start = end
            
    return dist

def brute_force_tsp(stations):
    max_dist = 1e12
    best = None

    for perm in permutations(stations):
        perm = list(perm)
        perm.append(perm[0])

        dist = total_distance(perm)
        if dist < max_dist:
            max_dist = dist
            best = perm
            
    return best

def subsets(iterable):
    ''' Return all subsets of iterable between length 2 and (len+1)/2
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3) '''
    s = list(iterable)
    return [[s for s in sub] for sub in chain.from_iterable(combinations(s, r) for r in range(2, ceil((len(s)+1)/2) ))]

def extract_stations(var_name, city: City):
    ''' x[1, 2] -> (Station1, Station2)
        x[1, 2, 1] -> (Station1, Station2, Rail1)
    '''
    i, j = var_name.index('['), var_name.index(']')
    indexes = [int(s_id) for s_id in var_name[i+1: j].split(',')]
    
    # If just two stations
    if len(indexes) == 2:
        return [city.get_station(s_id) for s_id in indexes]
    # Else 2 stations and rail
    return [city.get_station(s_id) for s_id in indexes[:2]] + [indexes[2]]
            
def base_pairs2rails(pairs, debug=False):
    ''' [(1, 2), (2, 3)] -> [1, 2, 3]'''

    # Step 1. Check if there is an endpoint
    starts = [pair[0] for pair in pairs]
    ends   = [pair[1] for pair in pairs]
    
    for end in ends:
        if end not in starts:
            station = end
            break
    else:
        station = starts[0]
        
    if debug: print(station)
    
    # Step 2. Build the rails
    dict_pairs = dict(pairs)
    rev_pairs = {v:k for (k,v) in dict_pairs.items()}
    
    # rails is a list that encodes the edge sequence
    rails = [station]
    while rev_pairs:
        if debug: print(rev_pairs)
        station = rev_pairs.pop(station)
        rails.append(station)
    
    return rails[::-1]

def pairs2rails(pairs, n_iter=20, debug=False):
    """Cheeky workaround; iterates pairs2rails `n_iter` times in case of picking wrong start"""
    from random import shuffle
    for _ in range(n_iter):
        try:
            shuffle(pairs)
            return base_pairs2rails(pairs, debug)
        except KeyError:
            continue
    else:
        raise KeyError()

#### SIMULATED ANNEALING #### 
def initialise(N, K):
    ''' Initialises N x N x K matrix `x` that creates a path between all stations '''
    x = np.zeros((N, N, K))
    
    order = list(range(N))
    
    for k in range(K):
        shuffle(order)
        for i in range(N-1):
            a = order[i]
            b = order[i+1]
            x[a, b, k] = x[b, a, k] = 1
            
    return x      

# Helper functions to make code readable
isolated_station = lambda x: not all(x.sum(axis=2).sum(axis=0))
rail_covers_every_station = lambda grid: not any(grid.sum(axis=0) == 0)
has_loop = lambda grid: not any(grid.sum(axis=0) == 1) and grid.sum() > 2
just_two_stations = lambda grid: (grid.sum() == 2)
get_stations_not_on_line = lambda grid: (grid.sum(axis=0) == 0).nonzero()[0]
get_middle_stations = lambda grid: (grid.sum(axis=0) == 2).nonzero()[0]
get_end_stations    = lambda grid: (grid.sum(axis=0) == 1).nonzero()[0]
get_line_stations   = lambda grid: (grid.sum(axis=0) != 0).nonzero()[0]
has_subtours = lambda grid: len(get_end_stations(grid)) > 2
empty = lambda grid: grid.sum() == 0

def get_neighbour(x, case=0):
    x = deepcopy(x)
    # How many lines to modify
    m = 1 + randint(x.shape[2])
    
    lines = list(range(x.shape[2]))
    shuffle(lines)
    
    lines_to_modify = lines[:m]
    
    for k in lines_to_modify:
        grid = x[:, :, k]
    
        pre_grid = deepcopy(grid)
    
        case = choice([1, 2, 3, 4]) if case == 0 else case
        
        if case == 1:
            # Case 1. Expand Rail line end (Includes forming loop)
            if not has_loop(grid):
                old_choices = list(get_end_stations(grid))
                new_choices = list(get_stations_not_on_line(grid)) + old_choices
                
                old = choice(old_choices)
                new = choice(new_choices)
                
                if new != old:
                    grid[old, new] = grid[new, old] = 1

        elif case == 2:
            # Case 2. Shrink Rail line end (Includes breaking loop)
            # This can make city invalid (considering station kinds)
            # But cost function makes the cost really high
            if has_loop(grid):
                # Pick edge to break
                choices = np.array(grid.nonzero()).T
                i, j = choices[choice(range(len(choices)))]
                grid[i, j] = grid[j, i] = 0
            else:
                # Pick end to shrink
                ends = get_end_stations(grid)
                old_end = choice(ends)
                new_end = grid[old_end, :].argmax()
                grid[old_end, new_end] = grid[new_end, old_end] = 0

                if isolated_station(x) or empty(grid):
                    grid[old_end, new_end] = grid[new_end, old_end] = 1

        elif case == 3:
            # Case 3. Add station in middle
            if not rail_covers_every_station(grid):
                choices = get_stations_not_on_line(grid)
                b = choice(choices)
                
                a = choice(get_line_stations(grid))
                c = choice(grid[a].nonzero()[0])
                
                grid[a, b] = grid[b, a] = grid[b, c] = grid[c, b] = 1
                grid[a, c] = grid[c, a] = 0

        elif case == 4:
            # Case 4. Remove station in middle
            # If not just 2 stations
            if not just_two_stations(grid):
                choices = get_middle_stations(grid)
                b = choice(choices)
                a, c = grid[b, :].nonzero()[0]
                grid[a, b] = grid[b, a] = grid[b, c] = grid[c, b] = 0
                prev_ac = grid[a, c]
                grid[a, c] = grid[c, a] = 1

                if isolated_station(x):
                    grid[a, b] = grid[b, a] = grid[b, c] = grid[c, b] = 1
                    grid[a, c] = grid[c, a] = prev_ac
        
    return x

def simulated_annealing(city, K, cost_fn, max_iter=100, cutoff_val=0.001, save=False, experiment_max_iters=[]):
    ''' Performs simulated annealing to find (try and find) the optimal configuration 
        K is number of lines '''
    # Initialise Configuration
    cur_x = initialise(len(city.stations), K)
    cur_cost = cost_fn(city, cur_x)
    best_x, best_cost = cur_x, cur_cost
    
    experiment_max_iters = set(experiment_max_iters)
    
    # Time
    t = 0
    
    cur_xs = []
    new_xs = []
    
    # Results for computational study
    costs = []
    t1s = []
    
    while True:
        if save:
            cur_xs.append(cur_x)
        
        # Update current temperature
        T = np.exp(np.log(cutoff_val)*t / max_iter)
        
        if t in experiment_max_iters:
            t1s.append(time())
            costs.append(best_cost)
            
        # End of simulated annealing
        if T < cutoff_val:
            if experiment_max_iters:
                return costs, t1s
            if save:
                return best_x, best_cost, cur_xs, new_xs
            # Return configuration and its value
            return best_x, best_cost
        
        # Make random change
        new_x = get_neighbour(cur_x)
        new_cost = cost_fn(city, new_x)
        
        if save:
            new_xs.append(new_x)
        
        # Keep change if it is an improvement (or randomly sometimes)
        if new_cost < cur_cost or np.random.uniform(0, 1) < T:
            cur_x, cur_cost = new_x, new_cost
            
            if cur_cost < best_cost:
                best_x, best_cost = cur_x, cur_cost
            
        t += 1
    
#### PLOTTING ####
def draw(s1: Station, s2: Station, ax, c: str, jitter=0) -> None:
    """Draws the (s1,s2)-rail connection, with `disp` offset for overlapping rails"""
    midx, midy = midpoint(s1, s2)
    
    # Straight
    jitter *= np.random.uniform(-0.005, 0.005, 4)
    ax.plot([s1.x+jitter[0], midx], 
            [s1.y+jitter[1], midy], lw=4, zorder=-1, c=c)
    
    # Bend
    ax.plot([midx, s2.x+jitter[2]], 
            [midy, s2.y+jitter[3]], lw=4, zorder=-1, c=c)


def graph(stations, rails, equal_aspect=False, show_station_ids=True, jitter=True):
    # Check if only one rail
    if isinstance(rails[0], Station):
        rails = [rails]
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    edges = dict()

    for i, rail in enumerate(rails):
        start = rail[0]
        for end in rail[1:]:
            draw(start, end, ax, f'C{i}', jitter)
            edges[(start, end)] = i
            start = end
    
    for i, s in enumerate(stations):
        ax.scatter(s.x, s.y, marker=s.shape, s=s.size, c='white', edgecolor='#382c27', linewidth=4)
        if show_station_ids:
            ax.text(s.x, s.y , i, fontsize=12, color="red", ha='center', va='center')

    fig.set_facecolor('#f7f7f5')
    ax.axis('off')
    
    if equal_aspect:
        fig.gca().set_aspect('equal')
    ax.set_xlim(min(s.x for s in stations)-0.1, max(s.x for s in stations) + 0.1)
    ax.set_ylim(min(s.y for s in stations)-0.1, max(s.y for s in stations) + 0.1)

    plt.show()

# Sorry Callum!
def graph_x(stations, x, equal_aspect=False, show_station_ids=True, jitter=True, filename=''):    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    edges = dict()

    for k in range(x.shape[2]):
        for i in range(x.shape[0]):
            for j in range(i+1, x.shape[0]):
                if x[i, j, k]:
                    start, end = stations[i], stations[j]
                    draw(start, end, ax, f'C{k}', jitter)
                    edges[(start, end)] = i
                    start = end
    
    for i, s in enumerate(stations):
        ax.scatter(s.x, s.y, marker=s.shape, s=s.size, c='white', edgecolor='#382c27', linewidth=4)
        if show_station_ids:
            ax.text(s.x, s.y , i, fontsize=12, color="red", ha='center', va='center')

    fig.set_facecolor('#f7f7f5')
    ax.axis('off')
    
    if equal_aspect:
        fig.gca().set_aspect('equal')
    ax.set_xlim(min(s.x for s in stations)-0.1, max(s.x for s in stations) + 0.1)
    ax.set_ylim(min(s.y for s in stations)-0.1, max(s.y for s in stations) + 0.1)

    if filename:
        plt.savefig('images/' + filename + '.png', facecolor='white', transparent=False, dpi=fig.dpi)
        plt.close()
    else:
        plt.show()
    
    return


#### DEPRECATED ####
def demand_by_centrality(stations, C=3):
    """Simple proxy demand for testing the framework
    # d = Cexp(-norm_2(rescaled(s - map_midpoint)))
    # Rescaled(s) = s' is s.t. norm_1(s') <= 1
    # And map_midpoint = midpoint/center of the minimal bounding box for the stations
    #
    # ASSUMES:
    # That more central stations (by how MM generates its worlds) get/need more traffic
    """
    import math
    lb = min([s.x for s in stations])
    rb = max([s.x for s in stations])
    db = min([s.y for s in stations])
    ub = max([s.y for s in stations])

    cx = (lb+rb)/2
    cy = (db+ub)/2
    d = np.zeros([len(stations)])
    for i, s in enumerate(stations):
        d[i] = C*math.exp(-math.sqrt(((s.x-cx)/(rb-lb))**2 + ((s.y-cy)/(ub-lb))**2))
    return d
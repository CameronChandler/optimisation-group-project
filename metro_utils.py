from __future__ import annotations
import numpy as np
from itertools import count, permutations, chain, combinations  
from math import ceil

class Station:
    __id_iter = count()
    all_stations = {}
    __locations = []
    SHAPE, SIZE = 0, 1
    KINDS = {'square': ['s', 500], 'circle': ['o', 500], 'triangle': ['^', 500], 
             'star': ['*', 700], 'plus': ['P', 500], 'pentagon': ['p', 500], 'diamond': ['D', 400]}
    HORIZONTAL, VERTICAL = 0, 1

    def __init__(self, x: float=np.random.uniform(), y: float=np.random.uniform(), kind: str='') -> None:
        self.x = x
        self.y = y
        # If kind not specified, pick randomly from options
        self.kind = np.random.choice(list(Station.KINDS.keys())) if not kind else kind
        
        assert(self.kind in Station.KINDS)
        
        self.shape = Station.KINDS[self.kind][Station.SHAPE]
        self.size  = Station.KINDS[self.kind][Station.SIZE]
        
        self.__set_id()
        
    def gen_stations(n_stations: int) -> None:
        ''' Creates list `stations` such that all stations are preferably greater than `threshold` apart '''
        Station.reset()
        
        [Station.__gen_dispersed_location() for station in range(n_stations)]

        return [Station(x, y) for x, y in Station.__locations]
    
    def __gen_dispersed_location(n_attempts: int=10) -> None:
        # Attempt to find good candidate up to n_attempts times
        for attempt in range(n_attempts):
            candidate = np.random.uniform(low=0, high=1, size=2)

            if Station.__invalid_candidate(candidate):
                continue

            else:
                break
                
        Station.__locations.append(candidate)
        
    def __set_id(self) -> None:
        self.id = next(Station.__id_iter)
        Station.all_stations[self.id] = self
        
    def get_station(id_: int) -> Station:
        return Station.all_stations[id_]
    
    def reset() -> None:
        Station.__id_iter = count()
        Station.all_stations = {}
        Station.__locations = []
        
    def __euclidean_dist(a: np.ndarray, b: np.ndarray) -> float:
        return np.linalg.norm(a - b)
    
    def __invalid_candidate(candidate: np.ndarray, threshold: float=0.15) -> bool:
        ''' If candidate is too close to any of the points - it is invalid '''
        return any(Station.__euclidean_dist(candidate, location) < threshold for location in Station.__locations) 
    
    def get_midpoint(s1: Station, s2: Station) -> tuple[float, float]:
        ''' Find the midpoint - the point where the bend is on the track between stations '''
        minn, maxx = Station.get_extreme_abs_difference(s1, s2)

        straight_len = maxx - minn
        
        Δx = s1.x - s2.x
        Δy = s1.y - s2.y

        alignment = Station.HORIZONTAL if abs(Δx) >= abs(Δy) else Station.VERTICAL
        dirx, diry = np.sign(Δx), np.sign(Δy)

        midx = s1.x - dirx * (alignment == Station.HORIZONTAL) * straight_len
        midy = s1.y - diry * (alignment == Station.VERTICAL  ) * straight_len

        return midx, midy
    
    def get_extreme_abs_difference(s1: Station, s2: Station) -> tuple[float, float]:
        lΔxl = abs(s1.x - s2.x)
        lΔyl = abs(s1.y - s2.y)

        return min(lΔxl, lΔyl), max(lΔxl, lΔyl)
    
    def distance(s1: Station, s2: Station) -> float:
        ''' Returns distance of rail between two adjacent stations '''
        minn, maxx = Station.get_extreme_abs_difference(s1, s2)
        return maxx + (2**0.5 - 1) * minn
        
    def __eq__(self, other):
        return self.id == other.id
        
    def __repr__(self):
        return str(self.id)
    
    def __hash__(self):
        return hash(self.id)

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

def extract_stations(var_name):
    ''' x[1, 2] -> (Station1, Station2)
        x[1, 2, 1] -> (Station1, Station2, Rail1)
    '''
    i, j = var_name.index('['), var_name.index(']')
    indexes = [int(s_id) for s_id in var_name[i+1: j].split(',')]
    
    # If just two stations
    if len(indexes) == 2:
        return [Station.get_station(s_id) for s_id in indexes]
    # Else 2 stations and rail
    return [Station.get_station(s_id) for s_id in indexes[:2]] + [indexes[2]]
            
def pairs2rails(pairs):
    ''' [(1, 2), (2, 3)] -> [1, 2, 3]'''

    # Step 1. Check if there is an endpoint
    starts = [pair[0] for pair in pairs]
    ends   = [pair[1] for pair in pairs]

    for end in ends:
        if end not in starts:
            station = end
            break
    else:
        # We have a cycle so starting point is arbitrary
        station = starts[0]

    # Step 2. Work through list building up
    pairs = dict(pairs)
    rails = [station]

    while pairs:
        station = pairs.pop(station)
        rails.append(station)
        
    return rails
    
#### PLOTTING ####

import matplotlib.pyplot as plt
import numpy as np

def draw(s1: Station, s2: Station, ax, c: str) -> None:
    
    midx, midy = Station.get_midpoint(s1, s2)
    
    # Straight
    ax.plot([s1.x, midx], 
            [s1.y, midy], lw=4, zorder=-1, c=c)
    
    # Bend
    ax.plot([midx, s2.x], [midy, s2.y], lw=4, zorder=-1, c=c)

def graph(stations, rails):
    # Check if only one rail
    if isinstance(rails[0], Station):
        rails = [rails]
    
    fig, ax = plt.subplots(figsize=(12, 12))

    for i, rail in enumerate(rails):
        start = rail[0]
        for end in rail[1:]:
            draw(start, end, ax, f'C{i}')
            start = end
    
    for s in stations:
        ax.scatter(s.x, s.y, marker=s.shape, s=s.size, c='white', edgecolor='#382c27', linewidth=4)

    fig.set_facecolor('#f7f7f5')
    ax.axis('off')

    ax.set_xlim(min(s.x for s in stations)-0.05, max(s.x for s in stations) + 0.05)
    ax.set_ylim(min(s.y for s in stations)-0.05, max(s.y for s in stations) + 0.05)

    plt.show()
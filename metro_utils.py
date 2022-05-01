from __future__ import annotations
import numpy as np
from itertools import permutations, chain, combinations  
from math import ceil
from station import Station
from city import City
import matplotlib.pyplot as plt

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
    
    midx, midy = midpoint(s1, s2)
    
    # Straight
    ax.plot([s1.x, midx], 
            [s1.y, midy], lw=4, zorder=-1, c=c)
    
    # Bend
    ax.plot([midx, s2.x], [midy, s2.y], lw=4, zorder=-1, c=c)

def graph(stations, rails, equal_aspect=False):
    # Check if only one rail
    if isinstance(rails[0], Station):
        rails = [rails]
    
    fig, ax = plt.subplots(figsize=(12, 12))

    for i, rail in enumerate(rails):
        start = rail[0]
        for end in rail[1:]:
            draw(start, end, ax, f'C{i}')
            start = end
    
    for i, s in enumerate(stations):
        ax.scatter(s.x, s.y, marker=s.shape, s=s.size, c='white', edgecolor='#382c27', linewidth=4)
        ax.text(s.x + 0.03, s.y + 0.03 , i, fontsize=12, color="red")

    fig.set_facecolor('#f7f7f5')
    ax.axis('off')
    
    if equal_aspect:
        fig.gca().set_aspect('equal')
    ax.set_xlim(min(s.x for s in stations)-0.05, max(s.x for s in stations) + 0.05)
    ax.set_ylim(min(s.y for s in stations)-0.05, max(s.y for s in stations) + 0.05)

    plt.show()
from __future__ import annotations
import numpy as np
from station import Station, KINDS

class City:
    
    def __init__(self, n_stations=None, locations=None, kinds=None):
        Station.reset()
        if n_stations is None and locations is None:
            raise ValueError("Can't have both n_stations and locations as `None`")
            
        self.n_stations = len(locations) if n_stations is None else n_stations     
        self._locations = self._gen_locations() if locations is None else locations
        self._kinds     = self._gen_kinds()     if kinds     is None else kinds
        self.stations = self.gen_stations()
        
    def gen_stations(self) -> None:
        ''' Create list of stations '''
        return [Station(x, y, kind) for (x, y), kind in zip(self._locations, self._kinds)]
        
    def _gen_locations(self) -> None:
        locations = []
        
        for _ in range(self.n_stations):
            loc = self._gen_dispersed_location(locations)
            locations.append(loc)
        
        return locations
    
    def _gen_dispersed_location(self, curr_locations: list[np.ndarray], n_attempts: int=10) -> None:
        # Attempt to find good candidate up to n_attempts times
        for attempt in range(n_attempts):
            candidate = np.random.uniform(low=0, high=1, size=2)

            if self._invalid_candidate(candidate, curr_locations):
                continue

            else:
                break
                
        return candidate
    
    def _invalid_candidate(self, candidate: np.ndarray, curr_locations: list, threshold: float=0.15) -> bool:
        ''' If candidate is too close to any of the points - it is invalid 
        '''
        return any(self._euclidean_dist(candidate, loc) < threshold for loc in curr_locations) 
        
    def _euclidean_dist(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.linalg.norm(a - b)
        
    def _gen_kinds(self) -> list[str]:
        taken_special = []
        kinds = []
        
        for _ in range(self.n_stations):
            options = [(kind, val[2]) for kind, val in KINDS.items() if kind not in taken_special]
            remaining_kinds = [e[0] for e in options]
            probs = np.array([e[1] for e in options])
            kind = np.random.choice(remaining_kinds, p=probs / probs.sum())
            kinds.append(kind)
            if kind in ['star', 'plus', 'pentagon', 'diamond']:
                taken_special.append(kind)
        
        return kinds
                                
    def get_station(self, id_: int) -> Station:
        ''' Return station with given `id_` '''
        return self.stations[id_]
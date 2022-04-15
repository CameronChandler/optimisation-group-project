from __future__ import annotations
import numpy as np
from station import Station, KINDS

class City:
    
    def __init__(self, n_stations: int, locations=None, kinds=None):
        Station.reset()
        self.n_stations = n_stations        
        self.__locations = self.__gen_locations() if locations is None else locations
        self.__kinds     = self.__gen_kinds()     if kinds     is None else kinds
        self.stations = self.gen_stations()
        
    def gen_stations(self) -> None:
        ''' Create list of stations '''
        return [Station(x, y, kind) for (x, y), kind in zip(self.__locations, self.__kinds)]
        
    def __gen_locations(self) -> None:
        locations = []
        
        for _ in range(self.n_stations):
            loc = self.__gen_dispersed_location(locations)
            locations.append(loc)
        
        return locations
    
    def __gen_dispersed_location(self, curr_locations: list[np.ndarray], n_attempts: int=10) -> None:
        # Attempt to find good candidate up to n_attempts times
        for attempt in range(n_attempts):
            candidate = np.random.uniform(low=0, high=1, size=2)

            if self.__invalid_candidate(candidate, curr_locations):
                continue

            else:
                break
                
        return candidate
    
    def __invalid_candidate(self, candidate: np.ndarray, curr_locations: list, threshold: float=0.15) -> bool:
        ''' If candidate is too close to any of the points - it is invalid '''
        return any(self.__euclidean_dist(candidate, loc) < threshold for loc in curr_locations) 
        
    def __euclidean_dist(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.linalg.norm(a - b)
        
    def __gen_kinds(self) -> list[str]:
        return list(np.random.choice(list(KINDS.keys()), self.n_stations))
                                
    def get_station(self, id_: int) -> Station:
        ''' Return station with given `id_` '''
        return self.stations[id_]
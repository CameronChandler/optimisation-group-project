from __future__ import annotations
import numpy as np
from itertools import count

SHAPE, SIZE = 0, 1
KINDS = {'square': ['s', 500], 'circle': ['o', 500], 'triangle': ['^', 500], 
         'star': ['*', 700], 'plus': ['P', 500], 'pentagon': ['p', 500], 'diamond': ['D', 400]}

class Station:
    __id_iter = count()

    def __init__(self, x: float=np.random.uniform(), y: float=np.random.uniform(), kind=None) -> None:
        self.x = x
        self.y = y
        # If kind not specified, pick randomly from options
        self.kind = self.__gen_kind() if kind is None else kind
        
        assert(self.kind in KINDS)
        
        self.shape = KINDS[self.kind][SHAPE]
        self.size  = KINDS[self.kind][SIZE]
        
        self.__set_id()
        
    def __gen_kind(self) -> str:
        return np.random.choice(list(KINDS.keys()))
        
    def __set_id(self) -> None:
        self.id = next(Station.__id_iter)
    
    def reset() -> None:
        Station.__id_iter = count()
        
    def __eq__(self, other):
        return self.id == other.id
        
    def __repr__(self):
        return str(self.id)
    
    def __hash__(self):
        return hash(self.id)
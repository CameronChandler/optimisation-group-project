from __future__ import annotations
import numpy as np
from itertools import count

SHAPE, SIZE = 0, 1
KINDS = {'square': ['s', 500, 0.1], 'circle': ['o', 500, 0.4], 'triangle': ['^', 500, 0.2], 
         'star': ['*', 700, 0.075], 'plus': ['P', 500, 0.075], 'pentagon': ['p', 500, 0.075], 'diamond': ['D', 400, 0.075]}

class Station:
    _id_iter = count()

    def __init__(self, x: float=np.random.uniform(), y: float=np.random.uniform(), kind=None, taken_special=[]) -> None:
        self.x = x
        self.y = y
        # If kind not specified, pick randomly from options
        self.kind = self._gen_kind(taken_special) if kind is None else kind
        
        assert(self.kind in KINDS)
        
        self.shape = KINDS[self.kind][SHAPE]
        self.size  = KINDS[self.kind][SIZE]
        
        self._set_id()
        
    def _gen_kind(self, taken_special) -> str:
        # Can't generate more than one of each special kind
        options = [(kind, val[2]) for kind, val in KINDS.items() if kind not in taken_special]
        remaining_kinds = [e[0] for e in options]
        probs = np.array([e[1] for e in options])

        return np.random.choice(remaining_kinds, p=probs / probs.sum())
        
    def _set_id(self) -> None:
        self.id = next(Station._id_iter)
    
    def reset() -> None:
        Station._id_iter = count()
        
    def __eq__(self, other):
        return self.id == other.id
        
    def __repr__(self):
        return f"<{str(self.id)}>"
    
    def __hash__(self):
        return hash(self.id)
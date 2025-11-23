from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class ShapeCostStats:
    shape: Tuple[int, ...]
    numel: int
    ndim: int
    samples: List[float]

    @property
    def avg_cost(self) -> float:
        return float(sum(self.samples) / len(self.samples))

    @property
    def sample_count(self) -> int:
        return len(self.samples)


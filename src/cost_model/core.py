import json
import math
from typing import Any, Dict, Optional, Tuple

from .data_types import ShapeCostStats
from .ingestion import ProfileIngester
from .regression import CostRegression
from .schedule_builder import ScheduleInputBuilder


class ConversionCostModel:
    """
    Learns a linear model: cost_ms ≈ α * numel + β * ndim + γ
    from conversion cost samples collected during profiling.
    """

    def __init__(self):
        self.shape_stats: Dict[Tuple[int, ...], ShapeCostStats] = {}
        self.regression = CostRegression()

    @property
    def fitted(self) -> bool:
        return self.regression.fitted

    @property
    def alpha(self) -> float:
        return self.regression.alpha

    @property
    def beta(self) -> float:
        return self.regression.beta

    @property
    def gamma(self) -> float:
        return self.regression.gamma

    @classmethod
    def from_profile_json(cls, path: str) -> "ConversionCostModel":
        """Load profiler output and train cost model."""
        with open(path, "r") as f:
            profile = json.load(f)

        model = cls()
        ProfileIngester.ingest_profile(profile, model.shape_stats)
        model.regression.fit(model.shape_stats)
        return model

    def estimate(self, shape: Tuple[int, ...]) -> Optional[float]:
        """Estimate conversion cost for a given shape."""
        shape = tuple(shape)

        if shape in self.shape_stats:
            return self.shape_stats[shape].avg_cost

        if self.regression.fitted:
            numel = int(math.prod(shape))
            ndim = len(shape)
            return self.regression.predict(numel, ndim)

        return None

    def build_schedule_input(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Build scheduler-ready input from profiler data."""
        return ScheduleInputBuilder.build(profile, self)


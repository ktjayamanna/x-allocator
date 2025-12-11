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

    def export_cost_json(self, profile_path: str, output_path: str):
        """
        Export cost.json for debugging purposes.

        Contains:
        - conversion_cost_table: measured .contiguous() costs
        - cost_model: regression coefficients (alpha, beta, gamma)

        Args:
            profile_path: Path to input profile.json
            output_path: Path to save cost.json
        """
        with open(profile_path, "r") as f:
            profile = json.load(f)

        cost_data = {
            "conversion_cost_table": profile.get("conversion_cost_table", {}),
            "cost_model": {
                "fitted": self.fitted,
                "alpha": self.alpha if self.fitted else None,
                "beta": self.beta if self.fitted else None,
                "gamma": self.gamma if self.fitted else None,
                "formula": "cost_ms ≈ alpha * numel + beta * ndim + gamma" if self.fitted else None
            }
        }

        with open(output_path, "w") as f:
            json.dump(cost_data, f, indent=2)

        print(f"Exported cost model data to {output_path}")

    def export_schedule_json(self, profile_path: str, output_path: str):
        """
        Export schedule.json for compiler.

        Contains:
        - ops: scheduler-ready operations list
        - gpu_idle_events: GPU idle events with op references

        Args:
            profile_path: Path to input profile.json
            output_path: Path to save schedule.json
        """
        with open(profile_path, "r") as f:
            profile = json.load(f)

        schedule_data = {
            "ops": self.build_schedule_input(profile)["ops"],
            "gpu_idle_events": profile.get("gpu_idle_events", [])
        }

        with open(output_path, "w") as f:
            json.dump(schedule_data, f, indent=2)

        print(f"Exported schedule data to {output_path}")


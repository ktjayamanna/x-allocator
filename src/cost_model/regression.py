import numpy as np
from typing import Dict, Tuple

from .data_types import ShapeCostStats


class CostRegression:
    """Fits linear regression model: cost_ms ≈ α * numel + β * ndim + γ"""

    def __init__(self):
        self.alpha = 0.0
        self.beta = 0.0
        self.gamma = 0.0
        self.fitted = False

    def fit(self, shape_stats: Dict[Tuple[int, ...], ShapeCostStats]):
        """Fit regression model from shape statistics."""
        if not shape_stats:
            print("WARNING: No conversion samples found. Regression skipped.")
            return

        X = []
        y = []

        for stats in shape_stats.values():
            if stats.sample_count == 0:
                continue
            X.append([stats.numel, stats.ndim, 1.0])
            y.append(stats.avg_cost)

        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)

        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        self.alpha, self.beta, self.gamma = coef.tolist()
        self.fitted = True

        print(f"Fitted cost model:")
        print(f"  cost_ms ≈ {self.alpha:.3e} * numel + "
              f"{self.beta:.3e} * ndim + {self.gamma:.3e}")

    def predict(self, numel: int, ndim: int) -> float:
        """Predict conversion cost for given numel and ndim."""
        if not self.fitted:
            return 0.0
        return self.alpha * numel + self.beta * ndim + self.gamma


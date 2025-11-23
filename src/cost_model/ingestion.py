import math
from typing import Any, Dict, List, Tuple

from .data_types import ShapeCostStats
from .utils import parse_shape


class ProfileIngester:
    """Ingests profiler data and extracts conversion cost samples."""

    @staticmethod
    def ingest_profile(
        profile: Dict[str, Any],
        shape_stats: Dict[Tuple[int, ...], ShapeCostStats]
    ):
        """Extract conversion cost samples from profiler output."""
        ProfileIngester._ingest_cost_table(profile, shape_stats)
        ProfileIngester._ingest_records(profile, shape_stats)

    @staticmethod
    def _ingest_cost_table(
        profile: Dict[str, Any],
        shape_stats: Dict[Tuple[int, ...], ShapeCostStats]
    ):
        """Extract samples from conversion_cost_table."""
        table = profile.get("conversion_cost_table", {})
        for shape_str, samples in table.items():
            shape = parse_shape(shape_str)
            samples = [float(s) for s in samples if s is not None]
            if samples:
                ProfileIngester._add_shape_samples(shape, samples, shape_stats)

    @staticmethod
    def _ingest_records(
        profile: Dict[str, Any],
        shape_stats: Dict[Tuple[int, ...], ShapeCostStats]
    ):
        """Extract samples from per-record raw_conversion_samples_ms."""
        for rec in profile.get("records", []):
            raw_samples = rec.get("raw_conversion_samples_ms", [])
            if not raw_samples:
                continue

            layouts = rec.get("input_layouts", []) + rec.get("output_layouts", [])
            if not layouts:
                continue

            main_layout = max(
                layouts,
                key=lambda l: int(math.prod(l["shape"])) if l.get("shape") else 0
            )
            shape = tuple(main_layout["shape"])
            samples = [float(s) for s in raw_samples if s is not None]

            if samples:
                ProfileIngester._add_shape_samples(shape, samples, shape_stats)

    @staticmethod
    def _add_shape_samples(
        shape: Tuple[int, ...],
        samples: List[float],
        shape_stats: Dict[Tuple[int, ...], ShapeCostStats]
    ):
        """Add samples for a given shape."""
        numel = int(math.prod(shape))
        ndim = len(shape)

        if shape not in shape_stats:
            shape_stats[shape] = ShapeCostStats(
                shape=shape, numel=numel, ndim=ndim, samples=[]
            )
        shape_stats[shape].samples.extend(samples)


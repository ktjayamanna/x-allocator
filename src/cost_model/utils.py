from typing import Tuple


def parse_shape(shape_str: str) -> Tuple[int, ...]:
    """Parse shape string like '(32, 128, 256)' into tuple."""
    cleaned = shape_str.strip().lstrip("(").rstrip(")")
    return tuple(int(x.strip()) for x in cleaned.split(",") if x.strip())


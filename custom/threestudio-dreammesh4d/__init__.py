import threestudio
from packaging.version import Version

if hasattr(threestudio, "__version__") and Version(threestudio.__version__) >= Version(
    "0.2.1"
):
    pass
else:
    if hasattr(threestudio, "__version__"):
        print(f"[INFO] threestudio version: {threestudio.__version__}")
    raise ValueError(
        "threestudio version must be >= 0.2.0, please update threestudio by pulling the latest version from github"
    )


from .background import gaussian_mvdream_background
from .geometry import (
    exporter,
    gaussian_base,
    gaussian_io,
    sugar,
    dynamic_sugar,
)
from .renderer import (
    diff_sugar_rasterizer_normal,
    diff_sugar_rasterizer_temporal,
)
from .system import (
    sugar_4dgen,
    sugar_static,
)
from .data import temporal_image, image
from .guidance import (
    temporal_stable_zero123_guidance,
)

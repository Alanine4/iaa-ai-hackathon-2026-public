"""C034: one source of truth for the two markets of Part 2.

developed: the Part 1 data (2021-2024); emerging: the Part 2 emerging-market data (2023-2024, test 2025).
The active market comes from the AIIAA_MARKET environment variable (default developed); scripts accept
`--market <name>`, which `select_from_argv` copies into that variable before the script resolves its paths.
Emerging outputs go to output/emerging/ so the Part 1 files in output/ are never overwritten.
"""
import os
from dataclasses import dataclass, field

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ENV = "AIIAA_MARKET"
DEFAULT = "developed"


@dataclass(frozen=True)
class Market:
    name: str
    folder: str  # under data/
    years: tuple  # training years
    out_subdir: str  # under output/; "" keeps the Part 1 location
    trend_share: float = None  # C019: share of the fitted log-linear trend (developed only)
    trends: dict = field(default_factory=dict)  # fixed per-line 2025 trend factors by option name (emerging)

    @property
    def data_dir(self):
        return os.path.join(ROOT, "data", *self.folder.split("/"))

    @property
    def out_dir(self):
        return os.path.join(ROOT, "output", self.out_subdir) if self.out_subdir else os.path.join(ROOT, "output")


MARKETS = {
    "developed": Market("developed", "iaa-ai-assisted-hackathon-2026-part-1", (2021, 2022, 2023, 2024), "",
                        trend_share=0.5),
    "emerging": Market(
        "emerging", "iaa-ai-assisted-hackathon-2026-part-2/emerging", (2023, 2024), "emerging",
        # C037: two training years (2024 not developed) cannot support a trend fit. Base run: no trend.
        # Probe: the developed market's half trend to 2025 (developed line_trend ** 0.5, C019).
        trends={"none": {"motor": 1.0, "buildings": 1.0, "contents": 1.0},
                "developed-half": {"motor": 1.0175, "buildings": 1.0948, "contents": 1.0636}}),
}


def get(name=None):
    """The named market, else the one in AIIAA_MARKET, else developed."""
    name = name or os.environ.get(ENV, DEFAULT)
    if name not in MARKETS:
        raise ValueError(f"Unknown market {name!r}; expected one of {sorted(MARKETS)}")
    return MARKETS[name]


def select_from_argv(argv):
    """Copy a `--market X` / `--market=X` option from argv into AIIAA_MARKET (argv itself is not changed)."""
    for i, arg in enumerate(argv):
        if arg == "--market" and i + 1 < len(argv):
            os.environ[ENV] = get(argv[i + 1]).name
        elif arg.startswith("--market="):
            os.environ[ENV] = get(arg.split("=", 1)[1]).name

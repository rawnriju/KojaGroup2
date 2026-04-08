import logging
import sys

# ---------------------------------------------------------------------------
# Standard Python logger for EnergyPlus simulation / env logging
# Used via:  from logger import logger
# Provides: logger.info(), logger.warning(), logger.error(), logger.exception()
# ---------------------------------------------------------------------------

def _setup_eplus_logger(name: str = "eplus_sim",
                        log_file: str | None = "eplus_sim.log",
                        level: int = logging.DEBUG) -> logging.Logger:
    """Create a standard Python logger that writes to console + optional file."""
    _logger = logging.getLogger(name)
    _logger.setLevel(level)

    if _logger.handlers:          # avoid duplicate handlers on re-import
        return _logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (INFO and above)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    _logger.addHandler(ch)

    # File handler (DEBUG and above) — captures everything
    if log_file:
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        _logger.addHandler(fh)

    return _logger


logger = _setup_eplus_logger()

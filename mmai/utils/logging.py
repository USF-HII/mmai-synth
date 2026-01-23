from rich.console import Console
from rich.logging import RichHandler
import logging

_console = Console()

def get_logger(name: str = "mmai"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=_console, markup=True, rich_tracebacks=True)],
    )
    return logging.getLogger(name)

def console():
    return _console

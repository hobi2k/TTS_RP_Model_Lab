try:
    from .charcard import generate_scenario_book
    from .generator import run_scenario
except ImportError:
    from charcard import generate_scenario_book
    from generator import run_scenario

__all__ = ["generate_scenario_book", "run_scenario"]

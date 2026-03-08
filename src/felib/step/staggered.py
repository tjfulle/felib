from .base import Step
from .static import StaticStep
from .heat_transfer import HeatTransferStep

from .base import Step
from .heat_transfer import HeatTransferStep
from .static import StaticStep


class StaggeredStep(Step):
    """
    Staggered (sequential) thermo-mechanical step.

    This step:
    1. Solves the thermal problem (HeatTransferStep)
    2. Then solves the mechanical problem (StaticStep)
    """

    def __init__(self, simulation, **kwargs):
        # Initialize parent Step class
        super().__init__(simulation, **kwargs)

        # Internally store the two sub-steps
        self.heat_step = HeatTransferStep(simulation, **kwargs)
        self.static_step = StaticStep(simulation, **kwargs)

    def run(self):
        # First solve temperature
        self.heat_step.run()

        # Then solve mechanics using updated temperatures
        self.static_step.run()
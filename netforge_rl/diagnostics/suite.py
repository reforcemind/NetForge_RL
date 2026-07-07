from netforge_rl.diagnostics.delayed_telemetry import DelayedTelemetry
from netforge_rl.diagnostics.false_positive import FalsePositiveRestraint
from netforge_rl.diagnostics.memory_probe import MemoryProbe
from netforge_rl.diagnostics.noisy_siem import NoisySIEM
from netforge_rl.diagnostics.ot_kinetic import OTKineticResponse
from netforge_rl.diagnostics.topology_shift import TopologyShift


def all_diagnostics():
    """The full capability-probe suite, one instance per capability."""
    return [
        MemoryProbe(),
        NoisySIEM(),
        DelayedTelemetry(),
        FalsePositiveRestraint(),
        OTKineticResponse(),
        TopologyShift(),
    ]

from cps.pythia import runner
from cps.pythia.native_state_packet import reconstruct_zero_adam_state


def test_runner_binds_packet_aware_native_reader_directly():
    assert runner.reconstruct_zero_adam_state is reconstruct_zero_adam_state

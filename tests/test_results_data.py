from app.results_data import ABLATE, REAL, SIM, SIM_BLIND, SOURCES


def test_sources_exist_in_logs():
    for path, needle in SOURCES:
        assert needle in open(path).read(), f"{needle} not found in {path}"


def test_ablation_blind_change_is_consistent():
    assert abs(SIM_BLIND[1][1] - SIM[3][1] - ABLATE[4][1]) < 0.001


def test_shapes():
    assert len(SIM) == 4 and len(REAL) == 2 and all(len(v) == 4 for v in REAL.values())

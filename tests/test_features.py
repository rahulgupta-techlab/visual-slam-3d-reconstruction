import numpy as np
from vo.features import match_descriptors


def test_ratio_matcher_returns_exact_matches():
    rng = np.random.default_rng(4)
    desc = rng.integers(0, 256, size=(20, 32), dtype=np.uint8)
    matches = match_descriptors(desc, desc.copy(), ratio=0.8)
    assert len(matches) == 20
    assert all(m.queryIdx == m.trainIdx for m in matches)


def test_missing_descriptors_return_empty():
    assert match_descriptors(None, None) == []

import numpy as np
from sai.stats.features import calc_fd, calc_df

def test_calc_fd_basic():
    # Basit örnek veri
    ref = np.array([[0, 1], [1, 1], [0, 0]])  # p1
    tgt = np.array([[0, 0], [1, 1], [1, 1]])  # p2
    src = np.array([[1, 1], [1, 1], [0, 0]])  # p3
    pos = np.array([100, 200, 300])

    score, loci = calc_fd(ref, tgt, [src], pos)

    assert isinstance(score, float)
    assert isinstance(loci, np.ndarray)
    assert loci.ndim == 1
    assert len(loci) > 0 or np.isnan(score)

def test_calc_fd_all_nan():
    # Tüm değerler aynı → fark = 0 → denominator = 0
    ref = np.ones((3, 2))  # p1
    tgt = np.ones((3, 2))  # p2
    src = np.ones((3, 2))  # p3
    pos = np.array([100, 200, 300])

    score, loci = calc_fd(ref, tgt, [src], pos)
    assert np.isnan(score)
    assert loci.size == 0

def test_calc_df_basic():
    ref = np.array([[0, 1], [1, 0], [0, 1]])
    tgt = np.array([[1, 1], [0, 0], [1, 0]])
    src = np.array([[1, 1], [0, 0], [1, 0]])
    pos = np.array([101, 202, 303])

    score, loci = calc_df(ref, tgt, [src], pos)
    assert isinstance(score, float)
    assert isinstance(loci, np.ndarray)
    assert loci.ndim == 1

def test_calc_df_div0_case():
    # src = tgt → p3 - p2 = 0 → denominator = 0
    ref = np.array([[0, 0], [1, 1]])
    tgt = np.array([[1, 1], [0, 0]])
    src = tgt.copy()
    pos = np.array([111, 222])

    score, loci = calc_df(ref, tgt, [src], pos)
    assert np.isnan(score)
    assert loci.size == 0


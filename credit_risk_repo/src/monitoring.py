import numpy as np

def psi(expected, actual, buckets=10):
    # population stability index between two distributions (arrays)
    def _pctile_cut(a, bins):
        return np.nanpercentile(a, np.linspace(0, 100, bins + 1))
    expected = np.array(expected)
    actual = np.array(actual)
    breaks = _pctile_cut(expected, buckets)
    eps = 1e-6
    psi_val = 0.0
    for i in range(len(breaks)-1):
        lb, ub = breaks[i], breaks[i+1]
        e_pct = ((expected >= lb) & (expected < ub)).sum() / (len(expected) + eps)
        a_pct = ((actual >= lb) & (actual < ub)).sum() / (len(actual) + eps)
        psi_val += (e_pct - a_pct) * np.log((e_pct + eps) / (a_pct + eps))
    return psi_val

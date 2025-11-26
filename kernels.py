import numpy as np

def h_rect_sym(t):
    """Prostokątny symetryczny [-0.5,0.5)."""
    return np.where((t >= -0.5) & (t < 0.5), 1.0, 0.0)

def h_tri(t):
    """Trójkątny kernel [-1,1]."""
    return np.where(np.abs(t) <= 1.0, 1 - np.abs(t), 0.0)

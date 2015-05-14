
import numpy as np

def compute_tx(constraints, err=1e-7):
	r = np.concatenate([np.roots(c) for c in constraints])
	r = r.real[abs(r.imag)<err]
	r = r[r>=-err]
	return np.min(r)

def compute_trajectory(d_l, j_l, a_l, v_l, x_l):
	t_d = compute_tx([[-d_l, j_l], [-d_l, 0, a_l], [-2*d_l, 0, 0, v_l], [-8*d_l, 0, 0, 0, x_l]])
	t_j = compute_tx([[-d_l*t_d, a_l - d_l*t_d**2], [-d_l*t_d, -3*d_l*t_d**2, -2*d_l*t_d**3 + v_l], [-2*d_l*t_d, -10*d_l*t_d**2, -16*d_l*t_d**3, -8*d_l*t_d**4 + x_l]])
	t_a = compute_tx([[-d_l*t_d**2 - d_l*t_d*t_j, -2*d_l*t_d**3 - 3*d_l*t_d**2*t_j - d_l*t_d*t_j**2 + v_l], [-d_l*t_d**2 - d_l*t_d*t_j, -6*d_l*t_d**3 - 9*d_l*t_d**2*t_j - 3*d_l*t_d*t_j**2, -8*d_l*t_d**4 - 16*d_l*t_d**3*t_j - 10*d_l*t_d**2*t_j**2 - 2*d_l*t_d*t_j**3 + x_l]])
	t_v = compute_tx([[-d_l*t_a*t_d**2 - d_l*t_a*t_d*t_j - 2*d_l*t_d**3 - 3*d_l*t_d**2*t_j - d_l*t_d*t_j**2, -d_l*t_a**2*t_d**2 - d_l*t_a**2*t_d*t_j - 6*d_l*t_a*t_d**3 - 9*d_l*t_a*t_d**2*t_j - 3*d_l*t_a*t_d*t_j**2 - 8*d_l*t_d**4 - 16*d_l*t_d**3*t_j - 10*d_l*t_d**2*t_j**2 - 2*d_l*t_d*t_j**3 + x_l]])
	t, d, j, a, v, x = 0, d_l, 0, 0, 0, 0
	yield t, d, j, a, v, x
	t, d, j, a, v, x = t + t_d, d_l, d_l*t_d + j, a + d_l*t_d**2/2 + j*t_d, a*t_d + d_l*t_d**3/6 + j*t_d**2/2 + v, a*t_d**2/2 + d_l*t_d**4/24 + j*t_d**3/6 + t_d*v + x
	yield t, d, j, a, v, x
	t, d, j, a, v, x = t + t_j, 0.0, j, a + j*t_j, a*t_j + j*t_j**2/2 + v, a*t_j**2/2 + j*t_j**3/6 + t_j*v + x
	yield t, d, j, a, v, x
	t, d, j, a, v, x = t + t_d, -d_l, -d_l*t_d + j, a - d_l*t_d**2/2 + j*t_d, a*t_d - d_l*t_d**3/6 + j*t_d**2/2 + v, a*t_d**2/2 - d_l*t_d**4/24 + j*t_d**3/6 + t_d*v + x
	yield t, d, j, a, v, x
	t, d, j, a, v, x = t + t_a, 0.0, j, a + j*t_a, a*t_a + j*t_a**2/2 + v, a*t_a**2/2 + j*t_a**3/6 + t_a*v + x
	yield t, d, j, a, v, x
	t, d, j, a, v, x = t + t_d, -d_l, -d_l*t_d + j, a - d_l*t_d**2/2 + j*t_d, a*t_d - d_l*t_d**3/6 + j*t_d**2/2 + v, a*t_d**2/2 - d_l*t_d**4/24 + j*t_d**3/6 + t_d*v + x
	yield t, d, j, a, v, x
	t, d, j, a, v, x = t + t_j, 0.0, j, a + j*t_j, a*t_j + j*t_j**2/2 + v, a*t_j**2/2 + j*t_j**3/6 + t_j*v + x
	yield t, d, j, a, v, x
	t, d, j, a, v, x = t + t_d, d_l, d_l*t_d + j, a + d_l*t_d**2/2 + j*t_d, a*t_d + d_l*t_d**3/6 + j*t_d**2/2 + v, a*t_d**2/2 + d_l*t_d**4/24 + j*t_d**3/6 + t_d*v + x
	yield t, d, j, a, v, x
	t, d, j, a, v, x = t + t_v, 0.0, j, a + j*t_v, a*t_v + j*t_v**2/2 + v, a*t_v**2/2 + j*t_v**3/6 + t_v*v + x
	yield t, d, j, a, v, x
	t, d, j, a, v, x = t + t_d, -d_l, -d_l*t_d + j, a - d_l*t_d**2/2 + j*t_d, a*t_d - d_l*t_d**3/6 + j*t_d**2/2 + v, a*t_d**2/2 - d_l*t_d**4/24 + j*t_d**3/6 + t_d*v + x
	yield t, d, j, a, v, x
	t, d, j, a, v, x = t + t_j, 0.0, j, a + j*t_j, a*t_j + j*t_j**2/2 + v, a*t_j**2/2 + j*t_j**3/6 + t_j*v + x
	yield t, d, j, a, v, x
	t, d, j, a, v, x = t + t_d, d_l, d_l*t_d + j, a + d_l*t_d**2/2 + j*t_d, a*t_d + d_l*t_d**3/6 + j*t_d**2/2 + v, a*t_d**2/2 + d_l*t_d**4/24 + j*t_d**3/6 + t_d*v + x
	yield t, d, j, a, v, x
	t, d, j, a, v, x = t + t_a, 0.0, j, a + j*t_a, a*t_a + j*t_a**2/2 + v, a*t_a**2/2 + j*t_a**3/6 + t_a*v + x
	yield t, d, j, a, v, x
	t, d, j, a, v, x = t + t_d, d_l, d_l*t_d + j, a + d_l*t_d**2/2 + j*t_d, a*t_d + d_l*t_d**3/6 + j*t_d**2/2 + v, a*t_d**2/2 + d_l*t_d**4/24 + j*t_d**3/6 + t_d*v + x
	yield t, d, j, a, v, x
	t, d, j, a, v, x = t + t_j, 0.0, j, a + j*t_j, a*t_j + j*t_j**2/2 + v, a*t_j**2/2 + j*t_j**3/6 + t_j*v + x
	yield t, d, j, a, v, x
	t, d, j, a, v, x = t + t_d, -d_l, -d_l*t_d + j, a - d_l*t_d**2/2 + j*t_d, a*t_d - d_l*t_d**3/6 + j*t_d**2/2 + v, a*t_d**2/2 - d_l*t_d**4/24 + j*t_d**3/6 + t_d*v + x
	yield t, d, j, a, v, x

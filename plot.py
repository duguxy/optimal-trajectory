from output import compute_trajectory
import numpy as np
import matplotlib.pyplot as plt

traj = np.array([x for x in compute_trajectory(0.1, 0.1, 1.0, 1.0, 10.0)])
for i in range(1, 5):
	plt.plot(traj[:, 0], traj[:, i])
plt.show()
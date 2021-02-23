import numpy as np
import matplotlib.pyplot as plt
from main import *

dt = 1

t = np.arange(0, len(training_data["t_mse_array"]), dt)

# Two signals with a coherent part at 10Hz and a random part
s1 = training_data["t_mse_array"]
s2 = training_data["v_mse_array"]

fig, axs = plt.subplots()
axs.plot(t, s2, label = 'Training MSE')
axs.plot(t, s1, label = 'Valid MSE')
axs.set_xlim(0, len(training_data["t_mse_array"]))
axs.set_xlabel('MSE')
axs.set_ylabel('KKKK')
axs.grid(True)


fig.tight_layout()
plt.show()
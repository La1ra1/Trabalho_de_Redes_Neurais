import numpy as np
from numpy import transpose
import matplotlib.pyplot as plt
from main import *

dt = 1

t = np.arange(0, len(training_data["t_mse_array"]), dt)

# Two signals with a coherent part at 10Hz and a random part
s1 = training_data["t_mse_array"]
s2 = training_data["v_mse_array"]

fig, axs = plt.subplots()

axs.plot(t, s1, label = 'Training MSE')
axs.plot(t, s2, label = 'Valid MSE')
axs.set_xlim(0, len(training_data["t_mse_array"]))
axs.set_xlabel('Épocas')
axs.set_ylabel('Erro Médio Quadrático')
axs.grid(True)



s4 = []

t2 = np.arange(0, len(results), dt)

for arr in output_validation:
    for number in arr:
        s4.append(number)    


fig, ax = plt.subplots()

ax.plot(t2, s4, label= 'Resultado do Dataset')
ax.plot(t2, results, label= 'Previsão da Rede')
ax.set_xlabel('Dados de Treino')
ax.set_ylabel('Valor do Resultado')
ax.grid(True)


#rects3 = ax[0].bar(len(s3[0]) + 0.35/3, transpose(s3[0][2]), width, label='Women')


fig.tight_layout()
plt.show()
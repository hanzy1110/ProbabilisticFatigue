import numpy as np
import matplotlib.pyplot as plt

dmgs = np.linspace(0,1.2, 1000)
delta_i = 1.25/19

aeran = np.array(1-np.power((1-dmgs),delta_i, dtype='complex'), dtype='complex')

fig, ax = plt.subplots(1, 1)
ax.plot(dmgs, np.abs(aeran), label="Aeran")
ax.plot(dmgs, dmgs, label = "linear")
ax.set_xlabel("Linear damage")
ax.set_ylabel("Aeran damage")

ax.legend()
plt.show()

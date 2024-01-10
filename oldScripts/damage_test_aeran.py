import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "ieee"])

dmgs = np.linspace(0, 1.0, 1000)
delta_i = 1.25 / 19

aeran = np.array(1 - np.power((1 - dmgs), delta_i, dtype="complex"), dtype="complex")

fig, ax = plt.subplots(1, 1)
ax.plot(dmgs, np.abs(aeran), label="Aeran")
ax.plot(dmgs, dmgs, label="Miner's Rule")
ax.set_xlabel(r"$\frac{n_i}{N_i}$")
ax.set_ylabel("Damage")

ax.legend()
plt.savefig("RESULTS/Aeran_model")

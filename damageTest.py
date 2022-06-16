import numpy as np
from src.damageModelGao import gaoModel

n_i = np.random.randint(10,1000, size=(10,))
N_i = np.random.randint(100000,1000000, size=(10,))

print(gaoModel(n_i, N_i))

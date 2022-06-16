import numpy as np
import matplotlib.pyplot as plt
from typing import List

def gaoModel(n_i:List[int], N_i:List[int]):

    totalN = len(n_i)-1
    def inner(k):
        print(k)
        if k == 0:
            return (1-n_i[k]/N_i[k])**(np.log(N_i[k+1])/np.log(N_i[k]))
        else:
            return (inner(k-1)-n_i[k]/N_i[k])**(np.log(N_i[k+1])/np.log(N_i[k]))

    return -1/(np.log(N_i[-1])) * (inner(totalN-1)-n_i[-1]/N_i[-1])


import numpy as np
import pandas as pd
from typing import List
from dataclasses import dataclass

@dataclass
class CableProps:
    Eal:float
    Ecore:float
    rw:float
    rc:float
    layers:List[int]
    T:float

    def getEImin(self):
        I_a = np.pi*self.rw**4/64
        I_c = np.pi*self.rc**4/64
        layers = np.array(self.layers)
        return self.Eal * layers[0] * I_a + self.Ecore * layers[1] * I_c

    def getLambda(self,T):
        self.lambda_ = np.sqrt(T/self.getEImin())

    def PFSlope(self, a:float=89.0):
        return self.Eal*(2*self.rw) * self.lambda_**2/(4*(np.exp(-self.lambda_*a)-1+self.lambda_*a))

from src.stressModel import CableProps, PFSlopeModel
import pandas as pd
import os

df = pd.read_csv(os.path.join('data', 'pfData.csv'))

props = {
    'Eal':69e3,
    'Ecore':207e3,
    'rw':1.86,'rc':1,
    'layers':[26,7],
    'T':97.4*1000*0.20
    # 'T':30641*0.20
         }

ydata = df[df['tension'] == 20]
print(ydata)
slopeModel = PFSlopeModel(ydata=ydata.values,
                          cableParams=props, 
                          resultPath='Results1')
slopeModel.restoreTrace()

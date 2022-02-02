#%%
import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict
import re

#%%

measures = {root:[] for root, _, _ in os.walk('csvs')}

for root, dirs, files in os.walk('csvs'):    
    for file in files:
        path = os.path.join(root, file)
        df = pd.read_csv(path)
        measures[root].append(df)

del measures['csvs']

# %%
aux ={}
for key, dfs in measures.items():
    aux[key] = []
    for df in dfs:
        aux[key].extend(list(df.keys()))

indexes = {}
pattern = r'^Fre'
for key, keys in aux.items():
    indexes[key] = list(filter(lambda x: re.search(pattern, x), keys))

pattern = r'\d+?'
for key, keys in aux.items():
    aux[key] = list(filter(lambda x: re.search(pattern, x), keys))
    
#%%  
concat_measures = {key: pd.concat([df for df in dfs],axis=1) for key, dfs in measures.items()}

INDEXES = {exp: concat_measures[exp][idx[0]] for exp, idx in indexes.items()}
INDEXES = {exp: INDEXES[exp].iloc[:,0] for exp in INDEXES.keys()}

series:Dict[str,pd.DataFrame] = {}

for key, keys in aux.items():
    series[key] = {key_: concat_measures[key][key_] for key_ in keys}
    series[key] = pd.DataFrame(series[key])
    series[key] = series[key].set_index(INDEXES[key])
# %%
for key,df in series.items():
    path = key.split('/')[1] + '.csv'
    df.to_csv(os.path.join('cleansed_csvs', path))
# %%

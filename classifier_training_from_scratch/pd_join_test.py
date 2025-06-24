#%%
# SETUP
import pandas as pd
import numpy as np

#%%
df1 = pd.DataFrame({"A": [1,2,3], "B":[5,6,7]})
df2 = pd.DataFrame({"A": [10,20,30], "B":[50,60,70]})


df_new = pd.concat([df1, df2])

import random

ind = np.random.choice(a=range(df_new.shape[0]), size=round(0.6*df_new.shape[0]), replace=False)
#random.choices(population=range(2), k=df_new)

df_new[-ind,:]

# %%

s1 = {i for i in range(4)}
s2 = {1,2,3,5}
s1-s2

ind = list(s1)
print(ind)
a = [1,2,3,4,5,6,7]
a[i for i in range(3)]
# %%

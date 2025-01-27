import numpy as np
import pandas as pd

def unique(series, ascending=False):
    '''Returns the unique values, along with the corresponding counts for a series'''

    # If at least one element is str, then the others should be treated as str as well.
    # The others might be nan for example, which does not play nicely with str.
    if str in set(type(i) for i in series):
        series = [str(i) for i in series]
    
    # Count frequencies
    counts = np.unique(series, return_counts=True)
    df = pd.DataFrame({"Values": counts[0], "Counts":counts[1]})
    df.sort_values(inplace=True, by="Counts", ascending=ascending)
    return df



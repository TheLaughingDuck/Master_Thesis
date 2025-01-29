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


# Combine various mr sequence types
def combine_mr_seqs(series):
    '''Takes a pandas series of types of mr sequences (str), and combines them according
    to provided instructions from domain expert.'''

    new_sequences = []

    for seq_type in series:
        new_type = seq_type

        # Mark these for removal
        if seq_type in ["UNKNOWN", "T1W_FSPGR_GD", "T1W_FSPGR", "SWAN", "SWI", "ASL", "MAG", "PHE", "ANGIO", "Vs3D", "FD"]: new_type = "remove"
        
        # These are T1W
        elif seq_type in ["T1W_SE", "T1W"]: new_type = "T1W"

        # These are T1W-GD
        elif seq_type in ["T1W_SE_GD", "T1W_SE_GD", "T1W_GD", "T1W-GD"]: new_type = "T1W-GD"

        # These are T1W FLAIR
        elif seq_type in ["T1W_FL"]: new_type = "T1W_FLAIR"

        # These are T1W FLAIR GAD
        elif seq_type in ["T1W_FL_GD"]: new_type = "T1W_FLAIR_GD"

        # These are MPR (separate from T1)
        elif seq_type in ["T1W_MPRAGE", ]: new_type = "MPR"

        # These are T1W-MPRAGE_GD
        elif seq_type in ["T1W_MPRAGE_GD"]: new_type = "T1W_MPRAGE_GD"
        
        # These are T2W
        elif seq_type in ["FSE", "tse", "T2W"]: new_type = "T2W"

        # These are FLAIR
        elif seq_type in ["T2W_TRIM", "FLAIR", "T2W_FLAIR"]: new_type = "FLAIR"

        ## These are T2W_FLAIR
        #elif seq_type in []: new_type = "T2W_FLAIR"

        new_sequences.append(new_type)
    return new_sequences
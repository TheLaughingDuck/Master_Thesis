In order to be able to access these core utilities in a jupyter interactive notebook, you must copy the following lines, and
put them in the beginning of that py script or (might work for jupyter notebooks as well).


###########################################
import os
import sys

nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)

import core_utils
###########################################
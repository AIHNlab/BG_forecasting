import sys
import os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), 'CG-EGA'))
from cg_ega.cg_ega import CG_EGA
import pandas as pd
freq = 5
y_true, y_pred = np.load("CG-EGA/example.npy")
results = pd.DataFrame(data = np.c_[y_true.reshape(-1,1), y_pred.reshape(-1,1)], columns=["y_true","y_pred"])
cg_ega = CG_EGA(results, freq)
print("AP, BE, EP:", cg_ega.reduced())
cg_ega.plot(day=0)
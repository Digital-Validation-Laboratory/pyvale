import numpy as np
import pandas as pd

expNo = 97

blockFile = f"{expNo}_Block.npy"
coilFile = f"{expNo}_Coil.npy"

outputFile = f"{expNo}_DIC_Normals.csv"

blockData = pd.DataFrame(np.load(blockFile).T, columns=["Xn_block","Yn_block","Zn_block"])
coilData = pd.DataFrame(np.load(coilFile).T, columns=["Xn_coil","Yn_coil","Zn_coil"])

data = pd.concat([blockData, coilData], axis=1)

data.to_csv(outputFile)
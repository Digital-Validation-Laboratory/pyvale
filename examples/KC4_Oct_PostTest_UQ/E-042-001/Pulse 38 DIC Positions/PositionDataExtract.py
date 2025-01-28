import numpy as np
import pandas as pd

xBlock = np.load("XBlock.npy")
yBlock = np.load("YBlock.npy")
zBlock = np.load("ZBlock.npy")

xCoil = np.load("XCoil.npy")
yCoil = np.load("YCoil.npy")
zCoil = np.load("ZCoil.npy")

print("Loaded data")

xBlockAv = np.mean(xBlock)
yBlockAv = np.mean(yBlock)
zBlockAv = np.mean(zBlock)

xCoilAv = np.mean(xCoil)
yCoilAv = np.mean(yCoil)
zCoilAv = np.mean(zCoil)

print(xBlockAv)
print(yBlockAv)
print(zBlockAv)

print(xCoilAv)
print(yCoilAv)
print(zCoilAv)

print(xBlockAv-xCoilAv)
print(yBlockAv-yCoilAv)
print(zBlockAv-zCoilAv)

print(np.std(yBlock-yCoil))
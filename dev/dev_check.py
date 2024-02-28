import numpy as np
import matplotlib.pyplot as plt


fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(0, 10, 0.1)
y = np.sin(x)
z = np.cos(x)

ax.plot(y, color='blue', label='Sine wave')
ax.plot(z, color='black', label='Cosine wave')
ax.set_title('Sine and cosine waves', fontsize=20)
ax.set_xlabel('Time', fontsize=16)
ax.set_ylabel('Intensity', fontsize=16)
leg = ax.legend()

plt.show()

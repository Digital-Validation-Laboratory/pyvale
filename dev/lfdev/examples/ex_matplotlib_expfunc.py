import numpy as np
import matplotlib.pyplot as plt
from pyvale import PlotProps

t_end = 4
t = np.linspace(0.01,t_end,100)
#f =  10*(1 - 2**-t)
a = 10
k = 0.5
f = a*(1 - np.exp(-(1/k)*t))

pp = PlotProps()
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

fig, ax = plt.subplots(figsize=pp.single_fig_size,layout='constrained')
fig.set_dpi(pp.resolution)

ax.plot(t,f,'-x',
    lw=pp.lw,ms=pp.ms,color=colors[0])

ax.set_xlabel(r'Time, $t$ [s]',
            fontsize=pp.font_ax_size, fontname=pp.font_name)
ax.set_ylabel(r'Temperature, $T$ [$\degree C$]',
            fontsize=pp.font_ax_size, fontname=pp.font_name)

#ax.set_xlim([np.min(p_time),np.max(p_time)])

plt.grid(True)
plt.draw()
plt.show()




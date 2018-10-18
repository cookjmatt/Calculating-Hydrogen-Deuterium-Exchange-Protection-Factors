import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from matplotlib import rc
from cycler import cycler

plt.style.use('fast')
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

example_peptide_one  = np.array([1, 5, 10, 0.01, 0.5, 0.2])
example_peptide_two  = np.array([0.1, 0.05, 0.005, 0.01, 0.125, 0.04])
example_peptide_pfs  = np.array([10, 100, 2000, 1, 4, 5])

def pf_plot(k1, k2, d_vals, length):
	x = np.logspace(-1.7,1.7, num=10000)
	y1 = length-np.exp(-k1*x[:,np.newaxis]).sum(axis=1)
	y2 = length-np.exp(-k2*x[:,np.newaxis]).sum(axis=1)
	d1_indices = [np.argmax(y1>i) for i in d_vals]
	d2_indices = [np.argmax(y2>i) for i in d_vals]
	d1_x = np.array([x[i] for i in d1_indices])
	d1_y = np.array([y1[i] for i in d1_indices])
	d2_x = np.array([x[i] for i in d2_indices])
	d2_y = np.array([y2[i] for i in d2_indices])
	gm = stats.gmean(d2_x/d1_x)
	return [x,y1,y2,gm,d1_x,d1_y,d2_x,d2_y]

fig, ([ax1, ax2]) = plt.subplots(nrows=1, ncols=2, sharex=False, sharey='row', figsize=(10,6.18))

k1 = np.array([1,5,10,0.01,0.5,0.2])
k2 = np.array([0.1, 0.05, 0.005, 0.01, 0.125, 0.04])
d_vals = np.linspace(0.25,6.0,30, endpoint=False)

k1 = np.array([0.1, 0.2, 3.0])
k2 = np.array(3*[0.39])
length = 3.0
d_vals = [.5,1,1.5,2,2.5]

x,y1,y2,gm,d1_x,d1_y,d2_x,d2_y = pf_plot(k1, k2, d_vals, length)
print(gm)
ax1.semilogx(x, y1, label="{:8}".format("f(t) = d; k = 0.1, 0.2, 3.0"), color=colors[0], linewidth=3, linestyle='-', marker='', mfc='none', ms=10)
ax1.semilogx(d1_x, d1_y, label=None, color=colors[0], linewidth=3, linestyle='', marker='o', mfc='none', ms=10)
ax1.semilogx(x, y2, label="{:8}".format("g(t) = d; k = 0.39"), color=colors[1], linewidth=3, linestyle='-', marker='', mfc='none', ms=10)
ax1.semilogx(d2_x, d2_y, label=None, color=colors[1], linewidth=3, linestyle='', marker='o', mfc='none', ms=10)
ax2.plot((d2_x / d1_x), d_vals, label="GM = {:.2f}".format(gm), color='black', linewidth=3, linestyle='', marker='o', mfc='none', ms=10)
ax2.axvline(gm, color='black', linewidth=2, linestyle='--', marker='v', markevery=2)

ax1.legend(loc=2,fontsize=10)
ax2.legend(loc=1,fontsize=10)
ax1.grid(True,which='major',linestyle='--')
ax2.grid(True,which='major',linestyle='--')
ax1.set_xlabel("Labeling Time $log(t)$")
ax1.set_ylabel("Relative Deuteration ($d$)")
ax2.set_xlabel("$g^{-1}(d) / f^{-1}(d)$")
ax2.set_xlim(0,3.0)
fig.subplots_adjust(wspace=.05, hspace=.5)
fig.savefig("01_example_peptide.png", bbox_inches='tight')
plt.close()



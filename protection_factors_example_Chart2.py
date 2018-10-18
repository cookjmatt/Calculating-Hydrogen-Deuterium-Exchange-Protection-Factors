import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from cycler import cycler

# Visual plotting settings
plt.style.use('fast')
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

# Calculate Protection Factor for two peptides of equal length N
#  Peptides are represented by arrays of rate constants
def pf_plot(k1, k2, d_vals, length):
	x = np.logspace(-2,3, num=10000)
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

#Define peptide arrays, length, and sampling values
k1 = np.array([1,5,10,0.01,0.5,0.2])
k2 = np.array([0.1, 0.05, 0.005, 0.01, 0.125, 0.04])
length = 6
d_vals = np.linspace(0.25,6.0,30, endpoint=False)

#Calculate Protection Factors and return relavant data for plotting
x,y1,y2,gm,d1_x,d1_y,d2_x,d2_y = pf_plot(k1, k2, d_vals, length)
print(gm)

### Charts ###
#Create figure with two subplots on one row
fig, ([ax1, ax2]) = plt.subplots(nrows=1, ncols=2, sharex=False, sharey='row', figsize=(10,6.18))

#Plot HDX curves for two peptides on first axis
ax1.semilogx(x, y1, label="{:8}".format("f(t) = d; k = 1, 5, 10,\n  .01, .5, .2"), color=colors[0], linewidth=3, linestyle='-', marker='', mfc='none', ms=10)
ax1.semilogx(d1_x, d1_y, label=None, color=colors[0], linewidth=3, linestyle='', marker='o', mfc='none', ms=10)
ax1.semilogx(x, y2, label="{:8}".format("g(t) = d; k = .1, .05,\n  .005,.01, .125, .04"), color=colors[1], linewidth=3, linestyle='-', marker='', mfc='none', ms=10)
ax1.semilogx(d2_x, d2_y, label=None, color=colors[1], linewidth=3, linestyle='', marker='o', mfc='none', ms=10)

#Plot quotients and GM and second axis
ax2.plot((d2_x / d1_x), d_vals, label="GM = {:.2f}".format(gm), color='black', linewidth=3, linestyle='', marker='o', mfc='none', ms=10)
ax2.axvline(gm, color='black', linewidth=2, linestyle='--', marker='v', markevery=2)

#Figure settings and save
ax1.legend(loc=2,fontsize=10)
ax2.legend(loc=1,fontsize=10)
ax1.grid(True,which='major',linestyle='--')
ax2.grid(True,which='major',linestyle='--')
ax1.set_xlabel("Labeling Time $log(t)$")
ax1.set_ylabel("Relative Deuteration ($d$)")
ax2.set_xlabel("$g^{-1}(d) / f^{-1}(d)$")
#ax2.set_xlim(0,3.0)
fig.subplots_adjust(wspace=.05, hspace=.5)
fig.savefig("02_example_peptide.png", bbox_inches='tight')
plt.close()
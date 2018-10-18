import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean
plt.style.use('fast')
E3CK = pd.read_csv('./HDX_ActiveBook_E3CK.csv')
E3CKUb = pd.read_csv('./HDX_ActiveBook_E3CKUb.csv')
LRRE3CK = pd.read_csv('./HDX_ActiveBook_LRRE3CK.csv')
LRRE3CKUb = pd.read_csv('./HDX_ActiveBook_LRRE3CKUb.csv')
LRRE3CK_PKN1 = pd.read_csv('./HDX_ActiveBook_LRRE3CK_PKN1.csv')
LRRE3CKUb_PKN1 = pd.read_csv('./HDX_ActiveBook_LRRE3CKUb_PKN1.csv')

curr = LRRE3CKUb_PKN1

for i in range(0, len(curr.index)):
	peptide = curr.iloc[i,6]
	times = [3.0, 60.0, 3600.0, 72000.0]
	values = [(curr.iloc[i,14]+curr.iloc[i,15])/2.0,
			  (curr.iloc[i,16]+curr.iloc[i,17])/2.0,
			  (curr.iloc[i,18]+curr.iloc[i,19])/2.0,
			  (curr.iloc[i,20]+curr.iloc[i,21])/2.0]
	values_corr = [(curr.iloc[i,29]+curr.iloc[i,30])/2.0,
				   (curr.iloc[i,31]+curr.iloc[i,32])/2.0,
				   (curr.iloc[i,33]+curr.iloc[i,34])/2.0,
				   (curr.iloc[i,35]+curr.iloc[i,36])/2.0,]
	errors = [curr.iloc[i,14]-values[0],
			  curr.iloc[i,16]-values[1],
			  curr.iloc[i,18]-values[2],
			  curr.iloc[i,20]-values[3]]
	errors_corr = [curr.iloc[i,29]-values_corr[0],
				   curr.iloc[i,31]-values_corr[1],
				   curr.iloc[i,33]-values_corr[2],
				   curr.iloc[i,35]-values_corr[3],]

	plt.figure()
	fig, ax1 = plt.subplots()
	ax1.set_xscale('log')
	ax1.errorbar(times, values, yerr=errors, fmt='b:o')
	ax1.tick_params('y', colors='b')
	ax1.set_title(peptide)
	ax1.set_xlabel('log[times(sec)]')
	ax1.set_ylabel('Relative Deuteration (Da)', color='b')
	ax1.set_ylim(ymin=0)
	ax2 = ax1.twinx()
	ax2.errorbar(times, values_corr, yerr=errors_corr, fmt='r:o')
	ax2.set_ylabel('% Calculated Total Deuteration', color='r')
	ax2.set_ylim(0.0,1.0)
	ax2.tick_params('y', colors='r')
	fig.tight_layout()
	filename = str("./LRRE3CKUb_PKN1_charts/LRRE3CKUb_PKN1_"+str(i)+"_"+peptide)
	print(filename)
	plt.savefig(filename, bbox_inches="tight")
	plt.close()
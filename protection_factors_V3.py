from os import sys
import math
import numpy as np
import pandas as pd
from scipy import stats
from scipy import interpolate
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
from matplotlib import rc
from cycler import cycler

### Matplotlib settings for graphing
plt.style.use('fast')
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
pastel1_cmap = cmx.get_cmap('Set2')

###

### Global Variables
time = np.array([3.0,60.0,1800.0,72000.0])	# Array of sampled timepoints in seconds
###

### Get the real residue numbers of a given peptide within the SspH1 sequence
###  Since peptide coverage is based on the E3CK construct MSMS data, that is the sequence used here
def E3CK_get_bounds(s):
    seq='GSHMASIRIHFDMAGPSVPREARALHLAVADWLTSAREGEAAQADRWQAFGLEDNAAAFSLVLDRLRETENFKKDAGFKAQISSWLTQLAEDAALRAKTFAMATEATSTCEDRVTHALHQMNNVQLVHNAEKGEYDNNLQGLVSTGREMFRLATLEQIAREKAGTLALVDDVEVYLAFQNKLKESLELTSVTSEMRFFDVSGVTVSDLQAAELQVKTAENSGFSKWILQWGPLHSVLERKVPERFNALREKQISDYEDTYRKLYDEVLKSSGLVDDTDAERTIGVSAMDSAKKEFLDGLRALVDEVLGSYLTARWRLN'
    l=seq.find(s)+383
    h=l+len(s)-1
    return (np.array([l,h]))
###

### Perform Log-linear interpolation with the alogrithm in Walters et al.
###  NOTE: I found the scipy algorithm to run better so this function is not actively being used
def log_lin(tx_plus,tx_minus,dx_plus,dx_minus,x):
    q1 = math.log10(tx_plus / tx_minus)
    d1 = dx_plus - dx_minus
    d2 = x - dx_minus

    r = 10**(((q1 / d1) * d2) + q1)
    if (r):
        return r
    else:
        return 1
###

### Calculate protection factor
def calc_pf(t, d1_in, d2_in, seq):

    d1 = np.copy(d1_in)
    d2 = np.copy(d2_in)
    error_return_default = [[0,0],[0,0],[0,0],[0,0],-1.0]

    # Calculate range of uptake values shared between d1 and d2 
    r = min(max(d1),max(d2))-max(min(d1),min(d2))
    if (r < 0):
        print("uptake range error with {:}".format(seq))
        return error_return_default

    # If a sampled HDX point has lower fractional deuteration than that preceding point, raise the deuteration point to the value
    #  of the preceding point. This ensures proper sampling points for interpolation. This does not change the raw data, it is only
    #  for picking sampling points (see examples)
    for i in range(0,len(d1)-1):
        if (d1[i+1] < d1[i]):
            d1[i+1] = d1[i]
        if (d2[i+1] < d2[i]):
            d2[i+1] = d2[i]

    # Pick sampling times and perform log-linear interpolation to obtain HDX values, Algorithm from Walters et al.
    d_prime = np.linspace(max(min(d1),min(d2)) + 0.001, min(max(d1),max(d2)) - 0.001,10*r)
    if (len(d_prime) < 10):
    	print("number of sampled points error with {:}".format(seq))
    	return error_return_default
    
    ### Log-linear interpolation algorithm from Walters et al. 
    ## NOTE: I found the scipy algorithm to run better so this function is not actively being used
    #t1_i_plus = [np.argmax(d1 > i) for i in d_prime]
    #t1_i_minus = np.maximum([(np.argmax(d1 > i) - 1) for i in d_prime], 0)
    #t2_i_plus = [np.argmax(d2 > i) for i in d_prime]
    #t2_i_minus = np.maximum([(np.argmax(d2 > i) -1) for i in d_prime], 0)
    #t1_prime = np.array([log_lin(t[t1_i_plus[i]],t[t1_i_minus[i]],d1[t1_i_plus[i]],d1[t1_i_minus[i]],d_prime[i]) for i in range(0,len(t1_i_minus))])
    #t2_prime = np.array([log_lin(t[t2_i_plus[i]],t[t2_i_minus[i]],d2[t2_i_plus[i]],d2[t2_i_minus[i]],d_prime[i]) for i in range(0,len(t2_i_minus))])
   
    # Log-linear interpolation from scipy, this works better than the algorithm above
    f1 = interpolate.interp1d(d1, np.log10(time))
    f2 = interpolate.interp1d(d2, np.log10(time))
    try:
    	t1_prime = 10**f1(d_prime)
    	t2_prime = 10**f2(d_prime)
    except:
    	print("interpolation error with {:}".format(seq))
    	return error_return_default

    # Calculate the protection factor
    quotient = t2_prime / t1_prime
    pf = stats.gmean(quotient)
    if (np.isnan(pf)):
    	print("pf calculation returns NAN error with {:}".format(seq))
    	pf = -1.0

    return [d_prime, t1_prime, t2_prime, quotient, pf]
###

### Get HDX in AciveBook format from CSV files ###
def get_data():
	# Read in a Pandas dataframe for each construct CSV saved from Activebook
    E3CK = pd.read_csv('./Activebooks/HDX_ActiveBook_E3CK_V2.csv')
    E3CKUb = pd.read_csv('./Activebooks/HDX_ActiveBook_E3CKUb_V2.csv')
    LRRE3CK = pd.read_csv('./Activebooks/HDX_ActiveBook_LRRE3CK_V2.csv')
    LRRE3CKUb = pd.read_csv('./Activebooks/HDX_ActiveBook_LRRE3CKUb_V2.csv')
    LRRE3CK_PKN1 = pd.read_csv('./Activebooks/HDX_ActiveBook_LRRE3CK_PKN1_V2.csv')
    LRRE3CKUb_PKN1 = pd.read_csv('./Activebooks/HDX_ActiveBook_LRRE3CKUb_PKN1_V2.csv')
    # Make lists of construct names and Pandas dataframe names
    names_list = ["E3CK", "E3CKUb", "LRRE3CK", "LRRE3CKUb", "LRRE3CK_PKN1", "LRRE3CKUb_PKN1"]
    filename_list = [E3CK, E3CKUb, LRRE3CK, LRRE3CKUb, LRRE3CK_PKN1, LRRE3CKUb_PKN1]
    # Make a dictionary for each construct
    d_E3CK={}
    d_E3CKUb={}
    d_LRRE3CK={}
    d_LRRE3CKUb={}
    d_LRRE3CK_PKN1={}
    d_LRRE3CKUb_PKN1={} 
    # Make a list of the construct dictionaries
    file_dict_list = [d_E3CK, d_E3CKUb, d_LRRE3CK, d_LRRE3CKUb, d_LRRE3CK_PKN1, d_LRRE3CKUb_PKN1]
    # Loop through each construct
    for file in range(len(filename_list)):
        curr = filename_list[file]
        # Loop through each peptide within the construct
        for i in range(0, len(curr)):
        	#Peptide name
            peptide = curr.iloc[i,6]
            # Mean of each sampled timepoint
            values = [(curr.iloc[i,14]+curr.iloc[i,15])/2.0,
                      (curr.iloc[i,16]+curr.iloc[i,17])/2.0,
                      (curr.iloc[i,18]+curr.iloc[i,19])/2.0,
                      (curr.iloc[i,20]+curr.iloc[i,21])/2.0]
            # Mean of each sampled timpoint corrected to fractional deuteration
            values_corr = [(curr.iloc[i,29]+curr.iloc[i,30])/2.0,
                           (curr.iloc[i,31]+curr.iloc[i,32])/2.0,
                           (curr.iloc[i,33]+curr.iloc[i,34])/2.0,
                           (curr.iloc[i,35]+curr.iloc[i,36])/2.0,]
            # Error of each sampled timepoint
            errors = [curr.iloc[i,14]-values[0],
                      curr.iloc[i,16]-values[1],
                      curr.iloc[i,18]-values[2],
                      curr.iloc[i,20]-values[3]]
            # Error of each sampled timepoint corrected to fractional deuteration
            errors_corr = [curr.iloc[i,29]-values_corr[0],
                            curr.iloc[i,31]-values_corr[1],
                           curr.iloc[i,33]-values_corr[2],
                           curr.iloc[i,35]-values_corr[3],]
            # For the current construct and peptide, add that peptide's HDX data to the dictionary
            #  for the construct in the list of constructs
            file_dict_list[file][peptide]=[values, values_corr, errors, errors_corr]
            
    # Get the list of total peptides, 'keys'
    keys = []
    for entry in file_dict_list[file]:
        keys.append(entry)

    # Make list of peptides containing HDX data
    peptides = []
    for i in range(0, len(E3CK)):
        data = []
        # Make a list of HDX data for each construct
        for file in range(len(filename_list)):
            peptide = file_dict_list[file][keys[i]]
            data.append(peptide)
        peptides.append(data)
    
    return keys, names_list, peptides
###

### Calculate Protection Factors ###
def SspH1_PF_Analysis():

	# Get data from helper function
    keys, constructs, peptides = get_data()

    # Arrays of protection factors for given comparisons
    #  There are 15 possible unique pairwise Protection Factor comparisons for the six constructs of SspH1
    #  Starting off with 5 interesting comparisons: Ub conjugation (3), addition of LRR, addition of PKN1 to LRR
    PF_E3CK_E3CKUb = []
    PF_LRRE3CK_LRRE3CKUb = []
    PF_LRRE3CK_PKN1_LRRE3CKUb_PKN1 = []
    PF_E3CK_LRRE3CK = []
    PF_E3CKUb_LRRE3CKUb = []
    PF_LRRE3CK_LRRE3CK_PKN1 = []
    PF_LRRE3CKUb_LRRE3CKUb_PKN1 = []

    # Arrays of all of the results of data analyis for the protection factors
    all_E3CK_E3CKUb = []
    all_LRRE3CK_LRRE3CKUb = []
    all_LRRE3CK_PKN1_LRRE3CKUb_PKN1 = []
    all_E3CK_LRRE3CK = []
    all_E3CKUb_LRRE3CKUb = []
    all_LRRE3CK_LRRE3CK_PKN1 = []
    all_LRRE3CKUb_LRRE3CKUb_PKN1 = []

    # Loop through each peptide and get data
    for key in range(0,len(keys)):
    	# Peptide sequence
        pep_name = keys[key]

        E3CK_name=constructs[0]
        E3CK_val=peptides[key][0][0]
        E3CK_valcorr=peptides[key][0][1]
        E3CK_err=peptides[key][0][2]
        E3CK_errcorr=peptides[key][0][3]
    
        E3CKUb_name=constructs[1]
        E3CKUb_val=peptides[key][1][0]
        E3CKUb_valcorr=peptides[key][1][1]
        E3CKUb_err=peptides[key][1][2]
        E3CKUb_errcorr=peptides[key][1][3]
    
        LRRE3CK_name=constructs[2]
        LRRE3CK_val=peptides[key][2][0]
        LRRE3CK_valcorr=peptides[key][2][1]
        LRRE3CK_err=peptides[key][2][2]
        LRRE3CK_errcorr=peptides[key][2][3]
    
        LRRE3CKUb_name=constructs[3]
        LRRE3CKUb_val=peptides[key][3][0]
        LRRE3CKUb_valcorr=peptides[key][3][1]
        LRRE3CKUb_err=peptides[key][3][2]
        LRRE3CKUb_errcorr=peptides[key][3][3]
    
        LRRE3CK_PKN1_name=constructs[4]
        LRRE3CK_PKN1_val=peptides[key][4][0]
        LRRE3CK_PKN1_valcorr=peptides[key][4][1]
        LRRE3CK_PKN1_err=peptides[key][4][2]
        LRRE3CK_PKN1_errcorr=peptides[key][4][3]
    
        LRRE3CKUb_PKN1_name=constructs[5]
        LRRE3CKUb_PKN1_val=peptides[key][5][0]
        LRRE3CKUb_PKN1_valcorr=peptides[key][5][1]
        LRRE3CKUb_PKN1_err=peptides[key][5][2]
        LRRE3CKUb_PKN1_errcorr=peptides[key][5][3]

        # Calculate protection factors for E3CK vs E3CKUb
        d_prime, t1_prime, t2_prime, quotient, pf = calc_pf(time, E3CK_val, E3CKUb_val, pep_name)
        PF_E3CK_E3CKUb.append(pf)
        all_E3CK_E3CKUb.append([d_prime, t1_prime, t2_prime, quotient, pf])

        # Calculate protection factors for LRRE3CK vs LRRE3CKUb
        d_prime, t1_prime, t2_prime, quotient, pf = calc_pf(time, LRRE3CK_val, LRRE3CKUb_val, pep_name)
        PF_LRRE3CK_LRRE3CKUb.append(pf)
        all_LRRE3CK_LRRE3CKUb.append([d_prime, t1_prime, t2_prime, quotient, pf])

        # Calculate protection factors for LRRE3CK_PKN1 vs LRRE3CKUb_PKN1
        d_prime, t1_prime, t2_prime, quotient, pf = calc_pf(time, LRRE3CK_PKN1_val, LRRE3CKUb_PKN1_val, pep_name)
        PF_LRRE3CK_PKN1_LRRE3CKUb_PKN1.append(pf)
        all_LRRE3CK_PKN1_LRRE3CKUb_PKN1.append([d_prime, t1_prime, t2_prime, quotient, pf])

        # Calculate protection factors for E3CK vs LRRE3CK
        d_prime, t1_prime, t2_prime, quotient, pf = calc_pf(time, E3CK_val, LRRE3CK_val, pep_name)
        PF_E3CK_LRRE3CK.append(pf)
        all_E3CK_LRRE3CK.append([d_prime, t1_prime, t2_prime, quotient, pf])

        # Calculate protection factors for E3CKUb vs LRRE3CKUb
        d_prime, t1_prime, t2_prime, quotient, pf = calc_pf(time, E3CKUb_val, LRRE3CKUb_val, pep_name)
        PF_E3CKUb_LRRE3CKUb.append(pf)
        all_E3CKUb_LRRE3CKUb.append([d_prime, t1_prime, t2_prime, quotient, pf])

        # Calculate protection factors for LRRE3CK vs LRRE3CK_PKN1
        d_prime, t1_prime, t2_prime, quotient, pf = calc_pf(time, LRRE3CK_val, LRRE3CK_PKN1_val, pep_name)
        PF_LRRE3CK_LRRE3CK_PKN1.append(pf)
        all_LRRE3CK_LRRE3CK_PKN1.append([d_prime, t1_prime, t2_prime, quotient, pf])

        # Calculate protection factors for LRRE3CKUb vs LRRE3CKUb_PKN1
        d_prime, t1_prime, t2_prime, quotient, pf = calc_pf(time, LRRE3CKUb_val, LRRE3CKUb_PKN1_val, pep_name)
        PF_LRRE3CKUb_LRRE3CKUb_PKN1.append(pf)
        all_LRRE3CKUb_LRRE3CKUb_PKN1.append([d_prime, t1_prime, t2_prime, quotient, pf])

    ### End main data analysis loop

    ### Make Protection Factor vs residue figures
    # Array of x values in mean residue number for each peptide within the full SspH1 sequence
    x = [np.mean(E3CK_get_bounds(keys[i])) for i in range(0, len(keys))]
    # PF Plot for E3CK vs E3CKUb
    fig1, ax1 = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, figsize=(10, 6.18))
    y1 = PF_E3CK_E3CKUb
    ax1.plot(x,y1,label="PFs for E3CK vs E3CKUb", color=colors[0], linewidth=3, linestyle='', marker='o', ms=7)  
    # PF Plot for LRRE3CK vs LRRE3CKUb
    fig2, ax2 = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, figsize=(10, 6.18))
    y2 = PF_LRRE3CK_LRRE3CKUb
    ax2.plot(x,y2,label="PFs for LRRE3CK vs LRRE3CKUb", color=colors[1], linewidth=3, linestyle='', marker='o', ms=7)  
    # PF Plot for LRRE3CK_PKN1 vs LRRE3CKUb_PKN1
    fig3, ax3 = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, figsize=(10, 6.18))
    y3 = PF_LRRE3CK_PKN1_LRRE3CKUb_PKN1
    ax3.plot(x,y3,label="PFs for LRRE3CK_PKN1 vs LRRE3CKUb_PKN1", color=colors[2], linewidth=3, linestyle='', marker='o', ms=7)  
    # PF Plot for E3CK vs LRRE3CK
    fig4, ax4 = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, figsize=(10, 6.18))
    y4 = PF_E3CK_LRRE3CK
    ax4.plot(x,y4,label="PFs for E3CK vs LRRE3CK", color=colors[3], linewidth=3, linestyle='', marker='o', ms=7)  
    # PF Plot for E3CKUb vs LRRE3CKUb
    fig5, ax5 = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, figsize=(10, 6.18))
    y5 = PF_E3CKUb_LRRE3CKUb
    ax5.plot(x,y5,label="PFs for E3CKUb vs LRRE3CKUb", color=colors[4], linewidth=3, linestyle='', marker='o', ms=7) 
    # PF Plot for LRRE3CK vs LRRE3CK_PKN1
    fig6, ax6 = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, figsize=(10, 6.18))
    y6 = PF_LRRE3CK_LRRE3CK_PKN1
    ax6.plot(x,y6,label="PFs for LRRE3CK vs LRRE3CK_PKN1", color=colors[5], linewidth=3, linestyle='', marker='o', ms=7)  
    # PF Plot for LRRE3CKUb vs LRRE3CKUb_PKN1
    fig7, ax7 = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, figsize=(10, 6.18))
    y7 = PF_LRRE3CKUb_LRRE3CKUb_PKN1
    ax7.plot(x,y7,label="PFs for LRRE3CKUb vs LRRE3CKUb_PKN1", color=colors[6], linewidth=3, linestyle='', marker='o', ms=7)
    # PF Plots for +/-Ub
    fig8, ax8 = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, figsize=(10, 6.18))
    ax8.plot(x,y1,label="PFs for E3CK vs E3CKUb", color=colors[0], linewidth=3, linestyle='', marker='o', ms=7)
    ax8.plot(x,y2,label="PFs for LRRE3CK vs LRRE3CKUb", color=colors[1], linewidth=3, linestyle='', marker='o', ms=7)
    ax8.plot(x,y3,label="PFs for LRRE3CK_PKN1 vs LRRE3CKUb_PKN1", color=colors[2], linewidth=3, linestyle='', marker='o', ms=7)
    # PF Plots for +/- LRR
    fig9, ax9 = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, figsize=(10, 6.18))
    ax9.plot(x,y4,label="PFs for E3CK vs LRRE3CK", color=colors[3], linewidth=3, linestyle='', marker='o', ms=7)
    ax9.plot(x,y5,label="PFs for E3CKUb vs LRRE3CKUb", color=colors[4], linewidth=3, linestyle='', marker='o', ms=7)
    # PF Plots for +/- PKN1
    fig10, ax10 = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, figsize=(10, 6.18))
    ax10.plot(x,y6,label="PFs for LRRE3CK vs LRRE3CK_PKN1", color=colors[5], linewidth=3, linestyle='', marker='o', ms=7)
    ax10.plot(x,y7,label="PFs for LRRE3CKUb vs LRRE3CKUb_PKN1", color=colors[6], linewidth=3, linestyle='', marker='o', ms=7)

    # Make array of Protection Factor axes for easy formatting
    ax_array = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10]
    for ax in ax_array:
    	ax.legend(fontsize=10)
    	ax.grid(True, which='major', linestyle='--')
    	ax.set_xlabel("Residue")
    	ax.set_ylabel("Protection Factor")
    # Save PF plot figures
    fig1.savefig("./temp/PF_1_E3CK_vs_E3CKUb.png", bbox_inches='tight')
    fig2.savefig("./temp/PF_2_LRRE3CK_vs_LRRE3CKUb.png", bbox_inches='tight')
    fig3.savefig("./temp/PF_3_LRRE3CK_PKN1_vs_LRRE3CKUb_PKN1.png", bbox_inches='tight')
    fig4.savefig("./temp/PF_4_E3CK_vs_LRRE3CK.png", bbox_inches='tight')
    fig5.savefig("./temp/PF_5_E3CKUb_vs_LRRE3CKUb.png", bbox_inches='tight')
    fig6.savefig("./temp/PF_6_LRRE3CK_vs_LRRE3CK_PKN1.png", bbox_inches='tight')
    fig7.savefig("./temp/PF_7_LRRE3CKUb_vs_LRRE3CKUb_PKN1.png", bbox_inches='tight')
    fig8.savefig("./temp/PF_8_+Ub", bbox_inches='tight')
    fig9.savefig("./temp/PF_9_+LRR", bbox_inches='tight')
    fig10.savefig("./temp/PF_10_+PKN1", bbox_inches='tight')
    plt.close()


    ### Individual Peptide Plots
    # Loop through each peptide and get data
    mean_res_numbers = [np.mean(E3CK_get_bounds(keys[i])) for i in range(0, len(keys))]
    peptide_length = [np.ptp(E3CK_get_bounds(keys[i]))+1 for i in range(0, len(keys))]
    for key in range(0,len(keys)):
        # Peptide sequence
        pep_name = keys[key]
        fig, ((a1, a2, a3, a4, a5),
              (a6, a7, a8, a9, a10),
              (a11, a12, a13, a14, a15),
              (a16, a17, a18, a19, a20),
              (a21, a22, a23, a24, a25)) = plt.subplots(nrows=5, ncols=5, sharex=False, sharey=False, figsize=(32, 20))

        ### E3CK vs E3CKUb Protection Factor Analysis
        # Remember: all_XX_.append([d_prime, t1_prime, t2_prime, quotient, pf])
        # E3CK HDX Plot
        a1.semilogx(time, peptides[key][0][0], label=constructs[0], color=colors[0], linewidth=3, linestyle='--', marker='o', ms=7)
        a1.grid(True, which='major', linestyle='--')
        a1.set_ylabel("Relative Deuteration ($d$)")
        a1.legend(fontsize=10)
        a1.set_ylim(ymin=0.0)
        # E3CKUb HDX Plot
        a2.semilogx(time, peptides[key][1][0], label=constructs[1], color=colors[1], linewidth=3, linestyle='--', marker='o', ms=7)
        a2.grid(True, which='major', linestyle='--')
        a2.legend(fontsize=10)
        a2.set_ylim(ymin=0.0)
        # E3CK and E3CKUb plots with sampling points for interpolation
        a3.semilogx(time, peptides[key][0][0], label="{:8}".format("f(t) = d"), color=colors[0], linewidth=3, linestyle='--', marker='', ms=10)
        a3.semilogx(all_E3CK_E3CKUb[key][1], all_E3CK_E3CKUb[key][0], colors[0], linewidth=3, linestyle='', marker='o', mfc='none', ms=10)
        a3.semilogx(time, peptides[key][1][0], label="{:8}".format("g(t) = d"), color=colors[1], linewidth=3, linestyle='--', marker='', ms=10)
        a3.semilogx(all_E3CK_E3CKUb[key][2], all_E3CK_E3CKUb[key][0], colors[1], linewidth=3, linestyle='', marker='o', mfc='none', ms=10)
        a3.grid(True, which='major', linestyle='--')
        a3.set_ylabel("Relative Deuteration ($d$)")
        a3.legend(fontsize=10)
        # E3CK and E3CKUb quotient of interpolated HDX points with vertical dashed Geometric Mean
        a4.plot(all_E3CK_E3CKUb[key][3], all_E3CK_E3CKUb[key][0], label="GM = {:.2f}".format(all_E3CK_E3CKUb[key][4]), color='black', linewidth=3, linestyle='', marker='o', mfc='none', ms=10)
        a4.axvline(all_E3CK_E3CKUb[key][4], color='black', linewidth=2, linestyle='--', marker='v', markevery=2)
        a4.grid(True, which='major', linestyle='--')
        a4.legend(fontsize=10)
        a4.set_ylim(a3.get_ylim())
        # PF Plot
        a5.plot(mean_res_numbers,PF_E3CK_E3CKUb,label="E3CK vs E3CKUb", color=pastel1_cmap(0.2), linewidth=3, linestyle='', marker='o', ms=7)
        a5.plot(mean_res_numbers[key],all_E3CK_E3CKUb[key][4],label=pep_name, color="red", linewidth=3, linestyle='', marker='o', ms=7)
        a5.legend(fontsize=10)
        a5.set_ylabel("Protection Factor")


        ### LRRE3CK vs LRRE3CKUb Protection Factor Analysis
        # LRRE3CK HDX Plot
        a6.semilogx(time, peptides[key][2][0], label=constructs[2], color=colors[2], linewidth=3, linestyle='--', marker='o', ms=7)
        a6.grid(True, which='major', linestyle='--')
        a6.set_ylabel("Relative Deuteration ($d$)")
        a6.legend(fontsize=10)
        a6.set_ylim(ymin=0.0)
        # LRRE3CKUb HDX Plot
        a7.semilogx(time, peptides[key][3][0], label=constructs[3], color=colors[3], linewidth=3, linestyle='--', marker='o', ms=7)
        a7.grid(True, which='major', linestyle='--')
        a7.legend(fontsize=10)
        a7.set_ylim(ymin=0.0)
        # LRRE3CK and LRRE3CKUb plots with sampling points for interpolation
        a8.semilogx(time, peptides[key][2][0], label="{:8}".format("f(t) = d"), color=colors[2], linewidth=3, linestyle='--', marker='', ms=10)
        a8.semilogx(all_LRRE3CK_LRRE3CKUb[key][1], all_LRRE3CK_LRRE3CKUb[key][0], colors[2], linewidth=3, linestyle='', marker='o', mfc='none', ms=10)
        a8.semilogx(time, peptides[key][3][0], label="{:8}".format("g(t) = d"), color=colors[3], linewidth=3, linestyle='--', marker='', ms=10)
        a8.semilogx(all_LRRE3CK_LRRE3CKUb[key][2], all_LRRE3CK_LRRE3CKUb[key][0], colors[3], linewidth=3, linestyle='', marker='o', mfc='none', ms=10)
        a8.grid(True, which='major', linestyle='--')
        a8.set_ylabel("Relative Deuteration ($d$)")
        a8.legend(fontsize=10)
        # LRRE3CK and LRRE3CKUb quotient of interpolated HDX points with vertical dashed Geometric Mean
        a9.plot(all_LRRE3CK_LRRE3CKUb[key][3], all_LRRE3CK_LRRE3CKUb[key][0], label="GM = {:.2f}".format(all_LRRE3CK_LRRE3CKUb[key][4]), color='black', linewidth=3, linestyle='', marker='o', mfc='none', ms=10)
        a9.axvline(all_LRRE3CK_LRRE3CKUb[key][4], color='black', linewidth=2, linestyle='--', marker='v', markevery=2)
        a9.grid(True, which='major', linestyle='--')
        a9.legend(fontsize=10)
        a9.set_ylim(a8.get_ylim())
        # PF Plot
        a10.plot(mean_res_numbers,PF_LRRE3CK_LRRE3CKUb,label="LRRE3CK vs LRRE3CKUb", color=pastel1_cmap(0.4), linewidth=3, linestyle='', marker='o', ms=7)
        a10.plot(mean_res_numbers[key],all_LRRE3CK_LRRE3CKUb[key][4],label=pep_name, color="red", linewidth=3, linestyle='', marker='o', ms=7)
        a10.legend(fontsize=10)
        a10.set_ylabel("Protection Factor")

        ### LRRE3CK_PKN1 vs LRRE3CKUb_PKN1 Protection Factor Analysis
        # LRRE3CK_PKN1 HDX Plot
        a11.semilogx(time, peptides[key][4][0], label=constructs[4], color=colors[4], linewidth=3, linestyle='--', marker='o', ms=7)
        a11.grid(True, which='major', linestyle='--')
        a11.set_ylabel("Relative Deuteration ($d$)")
        a11.legend(fontsize=10)
        a11.set_ylim(ymin=0.0)
        a11.set_xlabel("Labeling Time $log(t)$")
        # LRRE3CKUb_PKN1 HDX Plot
        a12.semilogx(time, peptides[key][5][0], label=constructs[5], color=colors[5], linewidth=3, linestyle='--', marker='o', ms=7)
        a12.grid(True, which='major', linestyle='--')
        a12.legend(fontsize=10)
        a12.set_ylim(ymin=0.0)
        # LRRE3CK_PKN1 and LRRE3CKUb_PKN1 plots with sampling points for interpolation
        a13.semilogx(time, peptides[key][4][0], label="{:8}".format("f(t) = d"), color=colors[4], linewidth=3, linestyle='--', marker='', ms=10)
        a13.semilogx(all_LRRE3CK_PKN1_LRRE3CKUb_PKN1[key][1], all_LRRE3CK_PKN1_LRRE3CKUb_PKN1[key][0], colors[4], linewidth=3, linestyle='', marker='o', mfc='none', ms=10)
        a13.semilogx(time, peptides[key][5][0], label="{:8}".format("g(t) = d"), color=colors[5], linewidth=3, linestyle='--', marker='', ms=10)
        a13.semilogx(all_LRRE3CK_PKN1_LRRE3CKUb_PKN1[key][2], all_LRRE3CK_PKN1_LRRE3CKUb_PKN1[key][0], colors[5], linewidth=3, linestyle='', marker='o', mfc='none', ms=10)
        a13.grid(True, which='major', linestyle='--')
        a13.set_ylabel("Relative Deuteration ($d$)")
        a13.legend(fontsize=10)
        # LRRE3CK_PKN1 and LRRE3CKUb-PKN1 quotient of interpolated HDX points with vertical dashed Geometric Mean
        a14.plot(all_LRRE3CK_PKN1_LRRE3CKUb_PKN1[key][3], all_LRRE3CK_PKN1_LRRE3CKUb_PKN1[key][0], label="GM = {:.2f}".format(all_LRRE3CK_PKN1_LRRE3CKUb_PKN1[key][4]), color='black', linewidth=3, linestyle='', marker='o', mfc='none', ms=10)
        a14.axvline(all_LRRE3CK_PKN1_LRRE3CKUb_PKN1[key][4], color='black', linewidth=2, linestyle='--', marker='v', markevery=2)
        a14.grid(True, which='major', linestyle='--')
        a14.legend(fontsize=10)
        a14.set_ylim(a13.get_ylim())
        # PF Plot
        a15.plot(mean_res_numbers,PF_LRRE3CK_PKN1_LRRE3CKUb_PKN1,label="LRRE3CK_PKN1 vs LRRE3CKUb_PKN1", color=pastel1_cmap(0.6), linewidth=3, linestyle='', marker='o', ms=7)
        a15.plot(mean_res_numbers[key],all_LRRE3CK_PKN1_LRRE3CKUb_PKN1[key][4],label=pep_name, color="red", linewidth=3, linestyle='', marker='o', ms=7)
        a15.legend(fontsize=10)
        a15.set_ylabel("Protection Factor")

        ### E3CK vs LRRE3CK Protection Factor Analysis
        # E3CK and LRRE3CK plots with sampling points for interpolation
        a18.semilogx(time, peptides[key][0][0], label="{:8}".format("f(t) = d"), color=colors[0], linewidth=3, linestyle='--', marker='', ms=10)
        a18.semilogx(all_E3CK_LRRE3CK[key][1], all_E3CK_LRRE3CK[key][0], colors[0], linewidth=3, linestyle='', marker='o', mfc='none', ms=10)
        a18.semilogx(time, peptides[key][2][0], label="{:8}".format("g(t) = d"), color=colors[2], linewidth=3, linestyle='--', marker='', ms=10)
        a18.semilogx(all_E3CK_LRRE3CK[key][2], all_E3CK_LRRE3CK[key][0], colors[2], linewidth=3, linestyle='', marker='o', mfc='none', ms=10)
        a18.grid(True, which='major', linestyle='--')
        a18.set_ylabel("Relative Deuteration ($d$)")
        a18.legend(fontsize=10)
        #Plot quotients and GM on axis to right
        a19.plot(all_E3CK_LRRE3CK[key][3], all_E3CK_LRRE3CK[key][0], label="GM = {:.2f}".format(all_E3CK_LRRE3CK[key][4]), color='black', linewidth=3, linestyle='', marker='o', mfc='none', ms=10)
        a19.axvline(all_E3CK_LRRE3CK[key][4], color='black', linewidth=2, linestyle='--', marker='v', markevery=2)
        a19.grid(True, which='major', linestyle='--')
        a19.legend(fontsize=10)
        a19.set_ylim(a18.get_ylim())
        # PF Plot
        a20.plot(mean_res_numbers,PF_E3CK_LRRE3CK,label="E3CK vs LRRE3CK", color=pastel1_cmap(0.8), linewidth=3, linestyle='', marker='o', ms=7)
        a20.plot(mean_res_numbers[key],all_E3CK_LRRE3CK[key][4],label=pep_name, color="red", linewidth=3, linestyle='', marker='o', ms=7)
        a20.legend(fontsize=10)
        a20.set_ylabel("Protection Factor")

        ### LRRE3CK and LRRE3CK_PKN1 Protection Factor Analysis
        # LRRE3CK HDX Plot
        # LRRE3CK and LRRE3CK_PKN1 plots with sampling points for interpolation
        a23.semilogx(time, peptides[key][2][0], label="{:8}".format("f(t) = d"), color=colors[2], linewidth=3, linestyle='--', marker='', ms=10)
        a23.semilogx(all_LRRE3CK_LRRE3CK_PKN1[key][1], all_LRRE3CK_LRRE3CK_PKN1[key][0], colors[2], linewidth=3, linestyle='', marker='o', mfc='none', ms=10)
        a23.semilogx(time, peptides[key][4][0], label="{:8}".format("g(t) = d"), color=colors[4], linewidth=3, linestyle='--', marker='', ms=10)
        a23.semilogx(all_LRRE3CK_LRRE3CK_PKN1[key][2], all_LRRE3CK_LRRE3CK_PKN1[key][0], colors[4], linewidth=3, linestyle='', marker='o', mfc='none', ms=10)
        a23.grid(True, which='major', linestyle='--')
        a23.set_ylabel("Relative Deuteration ($d$)")
        a23.legend(fontsize=10)
        a23.set_xlabel("Labeling Time $log(t)$")
        #Plot quotients and GM on axis to right
        a24.plot(all_LRRE3CK_LRRE3CK_PKN1[key][3], all_LRRE3CK_LRRE3CK_PKN1[key][0], label="GM = {:.2f}".format(all_LRRE3CK_LRRE3CK_PKN1[key][4]), color='black', linewidth=3, linestyle='', marker='o', mfc='none', ms=10)
        a24.axvline(all_LRRE3CK_LRRE3CK_PKN1[key][4], color='black', linewidth=2, linestyle='--', marker='v', markevery=2)
        a24.grid(True, which='major', linestyle='--')
        a24.legend(fontsize=10)
        a24.set_xlabel("$log[g^{-1}(d) / f^{-1}(d)]$")
        a24.set_ylim(a23.get_ylim())
        # PF Plot
        a25.plot(mean_res_numbers,PF_LRRE3CK_LRRE3CK_PKN1,label="LRRE3CK vs LRRE3CK_PKN1", color=pastel1_cmap(1.0), linewidth=3, linestyle='', marker='o', ms=7)
        a25.plot(mean_res_numbers[key],all_LRRE3CK_LRRE3CK_PKN1[key][4],label=pep_name, color="red", linewidth=3, linestyle='', marker='o', ms=7)
        a25.legend(fontsize=10)
        a25.set_ylabel("Protection Factor")
        a25.set_xlabel("Mean residue number")

        ### Overlay of all constructs
        a17.semilogx(time, peptides[key][0][0], label=constructs[0], color=colors[0], linewidth=3, linestyle='--', marker='o', ms=7)
        a17.semilogx(time, peptides[key][1][0], label=constructs[1], color=colors[1], linewidth=3, linestyle='--', marker='o', ms=7)
        a17.semilogx(time, peptides[key][2][0], label=constructs[2], color=colors[2], linewidth=3, linestyle='--', marker='o', ms=7)
        a17.semilogx(time, peptides[key][3][0], label=constructs[3], color=colors[3], linewidth=3, linestyle='--', marker='o', ms=7)
        a17.semilogx(time, peptides[key][4][0], label=constructs[4], color=colors[4], linewidth=3, linestyle='--', marker='o', ms=7)
        a17.semilogx(time, peptides[key][5][0], label=constructs[5], color=colors[5], linewidth=3, linestyle='--', marker='o', ms=7)
        a17.set_ylim(ymin=0.0)
        a17.grid(True, which='major', linestyle='--')
        a17.set_xlabel("Labeling Time $log(t)$")
        a17.set_ylabel("Relative Deuteration ($d$)")

        ### Plot mean residue number vs Protection Factors
        a16.plot(mean_res_numbers, PF_E3CK_E3CKUb, color=pastel1_cmap(0.2), linewidth=3, linestyle='', marker='o', ms=7)
        a16.plot(mean_res_numbers, PF_LRRE3CK_LRRE3CKUb, color=pastel1_cmap(0.4), linewidth=3, linestyle='', marker='o', ms=7)
        a16.plot(mean_res_numbers, PF_LRRE3CK_PKN1_LRRE3CKUb_PKN1, color=pastel1_cmap(0.6), linewidth=3, linestyle='', marker='o', ms=7)
        a16.plot(mean_res_numbers, PF_E3CK_LRRE3CK, color=pastel1_cmap(0.8), linewidth=3, linestyle='', marker='o', ms=7)
        a16.plot(mean_res_numbers, PF_LRRE3CK_LRRE3CK_PKN1, color=pastel1_cmap(1.0), linewidth=3, linestyle='', marker='o', ms=7)
        a16.plot(mean_res_numbers[key], PF_E3CK_E3CKUb[key], label=mean_res_numbers[key], color='red', linewidth=3, linestyle='', marker='o', ms=7)
        a16.plot(mean_res_numbers[key], PF_LRRE3CK_LRRE3CKUb[key], color='red', linewidth=3, linestyle='', marker='o', ms=7)
        a16.plot(mean_res_numbers[key], PF_LRRE3CK_PKN1_LRRE3CKUb_PKN1[key], color='red', linewidth=3, linestyle='', marker='o', ms=7)
        a16.plot(mean_res_numbers[key], PF_E3CK_LRRE3CK[key], color='red', linewidth=3, linestyle='', marker='o', ms=7)
        a16.plot(mean_res_numbers[key], PF_LRRE3CK_LRRE3CK_PKN1[key], color='red', linewidth=3, linestyle='', marker='o', ms=7)
        a16.set_ylabel("Protection Factor")
        a16.set_xlabel("Mean Residue Number")
        a16.legend(fontsize=10)

        ### Plot peptide length vs Protection Factors
        a21.plot(peptide_length, PF_E3CK_E3CKUb, color=pastel1_cmap(0.2), linewidth=3, linestyle='', marker='o', ms=7)
        a21.plot(peptide_length, PF_LRRE3CK_LRRE3CKUb, color=pastel1_cmap(0.4), linewidth=3, linestyle='', marker='o', ms=7)
        a21.plot(peptide_length, PF_LRRE3CK_PKN1_LRRE3CKUb_PKN1, color=pastel1_cmap(0.6), linewidth=3, linestyle='', marker='o', ms=7)
        a21.plot(peptide_length, PF_E3CK_LRRE3CK, color=pastel1_cmap(0.8), linewidth=3, linestyle='', marker='o', ms=7)
        a21.plot(peptide_length, PF_LRRE3CK_LRRE3CK_PKN1, color=pastel1_cmap(1.0), linewidth=3, linestyle='', marker='o', ms=7)
        a21.plot(peptide_length[key], PF_E3CK_E3CKUb[key], label=peptide_length[key], color='red', linewidth=3, linestyle='', marker='o', ms=7)
        a21.plot(peptide_length[key], PF_LRRE3CK_LRRE3CKUb[key], color='red', linewidth=3, linestyle='', marker='o', ms=7)
        a21.plot(peptide_length[key], PF_LRRE3CK_PKN1_LRRE3CKUb_PKN1[key], color='red', linewidth=3, linestyle='', marker='o', ms=7)
        a21.plot(peptide_length[key], PF_E3CK_LRRE3CK[key], color='red', linewidth=3, linestyle='', marker='o', ms=7)
        a21.plot(peptide_length[key], PF_LRRE3CK_LRRE3CK_PKN1[key], color='red', linewidth=3, linestyle='', marker='o', ms=7)
        a21.set_ylabel("Protection Factor")
        a21.set_xlabel("Peptide Length")
        a21.legend(fontsize=10)

        # Save full analysis figure for peptide
        #fig.subplots_adjust(wspace=.05, hspace=.5)
        bounds = E3CK_get_bounds(pep_name)
        supname= str("SspH1 "+str(bounds[0])+"-"+str(bounds[1])+" "+str(pep_name)+" ("+str(peptide_length[key])+" residues) - "+str(mean_res_numbers[key]))
        fig.suptitle(supname, fontsize=12, y=0.92)
        filename=str("./temp/"+str(bounds[0])+"-"+str(bounds[1])+"_"+pep_name+".png")
        fig.savefig(filename, bbox_inches='tight')
        print(filename)
        plt.close()
###

### Main
SspH1_PF_Analysis()
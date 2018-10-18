from os import sys
import math
import numpy as np
import pandas as pd
from scipy import stats
from scipy import interpolate
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from matplotlib import rc
from cycler import cycler

# Matplotlib settings
plt.style.use('fast')
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

example_peptide_one  = np.array([1, 5, 10, 0.01, 0.5, 0.2])
example_peptide_two  = np.array([0.1, 0.05, 0.005, 0.01, 0.125, 0.04])
example_peptide_pfs  = np.array([10, 100, 2000, 1, 4, 5])

time = np.array([3.0,60.0,1800.0,72000.0])

def get_data():
    E3CK = pd.read_csv('./HDX_ActiveBook_E3CK_V2.csv')
    E3CKUb = pd.read_csv('./HDX_ActiveBook_E3CKUb_V2.csv')
    LRRE3CK = pd.read_csv('./HDX_ActiveBook_LRRE3CK_V2.csv')
    LRRE3CKUb = pd.read_csv('./HDX_ActiveBook_LRRE3CKUb_V2.csv')
    LRRE3CK_PKN1 = pd.read_csv('./HDX_ActiveBook_LRRE3CKUb_PKN1_V2.csv')
    LRRE3CKUb_PKN1 = pd.read_csv('./HDX_ActiveBook_LRRE3CKUb_PKN1_V2.csv')
    names_list = ["E3CK", "E3CKUb", "LRRE3CK", "LRRE3CKUb", "LRRE3CK_PKN1", "LRRE3CKUb_PKN1"]
    filename_list = [E3CK, E3CKUb, LRRE3CK, LRRE3CKUb, LRRE3CK_PKN1, LRRE3CKUb_PKN1]
    d_E3CK={}
    d_E3CKUb={}
    d_LRRE3CK={}
    d_LRRE3CKUb={}
    d_LRRE3CK_PKN1={}
    d_LRRE3CKUb_PKN1={} 
    file_dict_list = [d_E3CK, d_E3CKUb, d_LRRE3CK, d_LRRE3CKUb, d_LRRE3CK_PKN1, d_LRRE3CKUb_PKN1]
    for file in range(len(filename_list)):
        curr = filename_list[file]
        for i in range(0, len(E3CK)):
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
            file_dict_list[file][peptide]=[values, values_corr, errors, errors_corr]
            
    # Get list of peptides, 'keys'
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

# Log-linear interpolation
def log_lin(tx_plus,tx_minus,dx_plus,dx_minus,x):
    q1 = math.log10(tx_plus / tx_minus)
    d1 = dx_plus - dx_minus
    d2 = x - dx_minus

    r = 10**(((q1 / d1) * d2) + q1)
    if (r):
        return r
    else:
        return 1

def calc_pf(t, d1_in, d2_in, seq):

    d1 = np.copy(d1_in)
    d2 = np.copy(d2_in)
    error_return_default = [[0,0],[0,0],[0,0],[0,0],0]

    # Calculate range of uptake values shared between d1 and d2 
    r = min(max(d1),max(d2))-max(min(d1),min(d2))
    if (r < 0):
        print("error with {:}".format(seq))
        return error_return_default

    for i in range(0,len(d1)-1):
        if (d1[i+1] < d1[i]):
            d1[i+1] = d1[i]
        if (d2[i+1] < d1[i]):
            d2[i+1] = d2[i]

    d_prime = np.linspace(max(min(d1),min(d2)) + 0.001, min(max(d1),max(d2)) - 0.001,10*r)
    t1_i_plus = [np.argmax(d1 > i) for i in d_prime]
    t1_i_minus = np.maximum([(np.argmax(d1 > i) - 1) for i in d_prime], 0)
    t2_i_plus = [np.argmax(d2 > i) for i in d_prime]
    t2_i_minus = np.maximum([(np.argmax(d2 > i) -1) for i in d_prime], 0)
    t1_prime = np.array([log_lin(t[t1_i_plus[i]],t[t1_i_minus[i]],d1[t1_i_plus[i]],d1[t1_i_minus[i]],d_prime[i]) for i in range(0,len(t1_i_minus))])
    t2_prime = np.array([log_lin(t[t2_i_plus[i]],t[t2_i_minus[i]],d2[t2_i_plus[i]],d2[t2_i_minus[i]],d_prime[i]) for i in range(0,len(t2_i_minus))])
    
    f1 = interpolate.interp1d(d1, np.log10(time))
    f2 = interpolate.interp1d(d2, np.log10(time))
    try:
    	t1_prime = 10**f1(d_prime)
    	t2_prime = 10**f2(d_prime)
    except:
    	pass

    quotient = t2_prime / t1_prime
    pf = stats.gmean(quotient)

    return [d_prime, t1_prime, t2_prime, quotient, pf]

def E3CK_get_bounds(s):
    seq='GSHMASIRIHFDMAGPSVPREARALHLAVADWLTSAREGEAAQADRWQAFGLEDNAAAFSLVLDRLRETENFKKDAGFKAQISSWLTQLAEDAALRAKTFAMATEATSTCEDRVTHALHQMNNVQLVHNAEKGEYDNNLQGLVSTGREMFRLATLEQIAREKAGTLALVDDVEVYLAFQNKLKESLELTSVTSEMRFFDVSGVTVSDLQAAELQVKTAENSGFSKWILQWGPLHSVLERKVPERFNALREKQISDYEDTYRKLYDEVLKSSGLVDDTDAERTIGVSAMDSAKKEFLDGLRALVDEVLGSYLTARWRLN'
    l=seq.find(s)+383
    h=l+len(s)-1
    return (np.array([l,h]))

def SspH1_PF_Analysis():

    keys, constructs, peptides = get_data()
    E3CK_vs_E3CKUb = []
    LRRE3CK_vs_LRRE3CKUb = []

    for key in range(1,len(keys)):
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

        d_prime, t1_prime, t2_prime, quotient, pf = calc_pf(time, E3CK_val, E3CKUb_val, pep_name)
        d_prime2, t1_prime2, t2_prime2, quotient2, pf2 = calc_pf(time, LRRE3CK_val, LRRE3CKUb_val, pep_name)
        E3CK_vs_E3CKUb.append([pep_name, pf])
        LRRE3CK_vs_LRRE3CKUb.append([pep_name, pf])

        individual_charts = False
        if (individual_charts):
            ### Charts ###
            #Create figure with two subplots on one row
            fig, ([ax1, ax2]) = plt.subplots(nrows=1, ncols=2, sharex=False, sharey='row', figsize=(10,6.18))

            #Plot HDX curves for two peptides on first axis
            ax1.semilogx(time, E3CK_val, label="{:8}".format("f(t) = d"), color=colors[0], linewidth=3, linestyle='-', marker='', mfc='none', ms=10)
            ax1.semilogx(t1_prime, d_prime, label=None, color=colors[0], linewidth=3, linestyle='', marker='o', mfc='none', ms=10)
            ax1.semilogx(time, E3CKUb_val, label="{:8}".format("g(t) = d"), color=colors[1], linewidth=3, linestyle='-', marker='', mfc='none', ms=10)
            ax1.semilogx(t2_prime, d_prime, label=None, color=colors[1], linewidth=3, linestyle='', marker='o', mfc='none', ms=10)

            #Plot quotients and GM and second axis
            ax2.plot(quotient, d_prime, label="GM = {:.2f}".format(pf), color='black', linewidth=3, linestyle='', marker='o', mfc='none', ms=10)
            ax2.axvline(pf, color='black', linewidth=2, linestyle='--', marker='v', markevery=2)

            #Figure settings and save
            ax1.legend(loc=2,fontsize=10)
            ax2.legend(loc=1,fontsize=10)
            ax1.grid(True,which='major',linestyle='--')
            ax2.grid(True,which='major',linestyle='--')
            ax1.set_xlabel("Labeling Time $log(t)$")
            ax1.set_ylabel("Relative Deuteration ($d$)")
            ax2.set_xlabel("$log[g^{-1}(d) / f^{-1}(d)]$")
            #ax2.set_xlim(0,3.0)
            fig.subplots_adjust(wspace=.05, hspace=.5)
            bounds = E3CK_get_bounds(pep_name)
            supname= str("SspH1 E3CK "+str(bounds[0])+" - "+str(bounds[1])+" "+str(pep_name))
            fig.suptitle(supname, fontsize=12, y=0.92)
            filename=str("./temp/"+"{:02d}".format(key)+"_"+pep_name+".png")
            print(filename)
            fig.savefig(filename, bbox_inches='tight')
            plt.close()

    PF_v_seq = True
    if (PF_v_seq):
        fig, ax = plt.subplots(nrows=1, ncols=1, sharex=False, sharey='row', figsize=(10, 6.18))
        x = [np.mean(E3CK_get_bounds(E3CK_vs_E3CKUb[i][0])) for i in range(0,len(E3CK_vs_E3CKUb))]
        y = [E3CK_vs_E3CKUb[i][1] for i in range(0,len(E3CK_vs_E3CKUb))]
        x2 = [np.mean(E3CK_get_bounds(E3CK_vs_E3CKUb[i][0])) for i in range(0,len(LRRE3CK_vs_LRRE3CKUb))]
        y2 = [E3CK_vs_E3CKUb[i][1] for i in range(0,len(LRRE3CK_vs_LRRE3CKUb))]
        ax.plot(x,y,label="E3CK vs E3CKUb", color="black", linewidth=3, linestyle='', marker='o', ms=7)
        #ax.semilogy(x2,y2,label="LRRE3CK vs LRRE3CKUb", color="red", linewidth=3, linestyle='', marker='o', ms=7)
        for i in range(0,len(x)):
            x1, y1 = [x[i],x[i]], [1,y[i]]
            x2, y2 = E3CK_get_bounds(E3CK_vs_E3CKUb[i][0]), [y[i],y[i]]
            plt.plot(x1, y1, x2, y2, color="black", linewidth=1, linestyle='--', marker='')
            print
            #x1, y1 = [x2[i],x2[i]], [1,y2[i]]
            #x2, y2 = E3CK_get_bounds(LRRE3CK_vs_LRRE3CKUb[i][0]), [y2[i],y2[i]]
            #plt.plot(x1, y1, x2, y2, color="red", linewidth=1, linestyle='--', marker='')
        ax.legend(fontsize=10)
        ax.grid(True,which='major',linestyle='--')
        ax.set_xlabel("Residue")
        ax.set_ylabel("Protection Factor")
        #ax.set_ylim(.5, 4)
        filename=str("./temp/"+"00_E3CK_vs_E3CKUb"+".png")
        fig.savefig(filename,bbox_inches='tight')
        print(filename)
        plt.close()

### Start of Analysis
SspH1_PF_Analysis()
sys.exit()

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
ax1.semilogx(x, y1, label="{:8}".format("f(t) = d"), color=colors[0], linewidth=3, linestyle='-', marker='', mfc='none', ms=10)
ax1.semilogx(d1_x, d1_y, label=None, color=colors[0], linewidth=3, linestyle='', marker='o', mfc='none', ms=10)
ax1.semilogx(x, y2, label="{:8}".format("g(t) = d"), color=colors[1], linewidth=3, linestyle='-', marker='', mfc='none', ms=10)
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
fig.savefig("03_example_peptide.png", bbox_inches='tight')
plt.close()
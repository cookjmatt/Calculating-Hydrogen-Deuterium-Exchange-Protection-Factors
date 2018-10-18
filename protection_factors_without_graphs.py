import numpy as np
import pandas as pd
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from cycler import cycler
import math
import bisect
from statistics import mean

# Settings and Global variables
plt.style.use('fast')
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
time = [3.0,60.0,1800.0,72000.0]

def get_data():
    E3CK = pd.read_csv('./HDX_ActiveBook_E3CK.csv')
    E3CKUb = pd.read_csv('./HDX_ActiveBook_E3CKUb.csv')
    LRRE3CK = pd.read_csv('./HDX_ActiveBook_LRRE3CK.csv')
    LRRE3CKUb = pd.read_csv('./HDX_ActiveBook_LRRE3CKUb.csv')
    LRRE3CK_PKN1 = pd.read_csv('./HDX_ActiveBook_LRRE3CK_PKN1.csv')
    LRRE3CKUb_PKN1 = pd.read_csv('./HDX_ActiveBook_LRRE3CKUb_PKN1.csv')
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
        for i in range(0, 53):
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
    for i in range(0,53):
        data = []
        # Make a list of HDX data for each construct
        for file in range(len(filename_list)):
            peptide = file_dict_list[file][keys[i]]
            data.append(peptide)
        peptides.append(data)
    
    return keys, names_list, peptides

def index_m(a, x):
    'Locate the leftmost value exactly equal to x'
    if (a[0] == x):
        return 0
    else:
        i = bisect.bisect_left(a, x)
        return i
    #if i != len(a) and a[i] == x:
    #    return i
    #raise ValueError

def find_lt(a, x):
    'Find rightmost value less than x'
    i = bisect.bisect_left(a, x)
    return a[i-1]
    #if i:
    #    return a[i-1]
    #raise ValueError

def find_gt(a, x):
    'Find leftmost value greater than x'
    i = bisect.bisect_right(a, x)
    if i != len(a):
        return a[i]
    else:
        return a[i-1]

# dci and tci are matched vectors that contain the amount of carried label and associated
#  experimental labeling time for each experimental data point, respectively
#
# calc_pf determines the empirical averaged protection factor between two HDX conditions for a given peptide 
def calc_pf(dc1, tc1, dc2, tc2):
    # Determine range of uptake values shared between dc1 and dc2
    l = max(min(dc1),min(dc2))
    h = min(max(dc1),max(dc2))
    r = h-l
    # Compare measurements 10 times for each observed unit of deuterium to exchange
    m = math.floor(10*r)
    if (m <= 1):
        return 0, [], [], [], [], [], [], [], [], [], [], [], []
    d_prime = [l + (r/(m+1))]
    for i in range(m-1):
        temp_var = d_prime[-1]
        d_prime.append(temp_var + (r/(m+1)))
        
    tc1_prime = []
    tc2_prime = []
    
    PF=1
 
    pfs = []
    dc1_plus = []
    dc1_minus = []
    dc2_plus = []
    dc2_minus = []
    tc1_plus = []
    tc1_minus = []
    tc2_plus = []
    tc2_minus = []

    for i in range(m):
        x = d_prime[i]
        
        dc1_x_plus = find_gt(dc1,x)
        dc1_plus.append(dc1_x_plus)
        dc1_x_plus_index = index_m(dc1,dc1_x_plus)
        print(dc1_x_plus_index, dc1, dc1_x_plus)
        dc1_x_minus = find_lt(dc1,x)
        dc1_minus.append(dc1_x_minus)
        dc1_x_minus_index = index_m(dc1,dc1_x_minus)
        tc1_x_plus = tc1[dc1_x_plus_index]
        tc1_plus.append(tc1_x_plus)
        tc1_x_minus = tc1[dc1_x_minus_index]
        tc1_minus.append(tc1_x_minus)
        
        dc2_x_plus = find_gt(dc2,x)
        dc2_plus.append(dc2_x_plus)
        dc2_x_plus_index = index_m(dc2,dc2_x_plus)
        dc2_x_minus = find_lt(dc2,x)
        dc2_minus.append(dc2_x_minus)
        dc2_x_minus_index = index_m(dc2,dc2_x_minus)
        tc2_x_plus = tc2[dc2_x_plus_index]
        tc2_plus.append(tc2_x_plus)
        tc2_x_minus = tc2[dc2_x_minus_index]
        tc2_minus.append(tc2_x_minus)
        
        tc1_prime.append(log_lin(tc1_x_plus,tc1_x_minus,dc1_x_plus,dc1_x_minus,x))
        tc2_prime.append(log_lin(tc2_x_plus,tc2_x_minus,dc2_x_plus,dc2_x_minus,x))
  
        PF *= (tc2_prime[i] / tc1_prime[i])
        pfs.append(PF)
   
    PF = PF**(1/m)
    return PF, pfs, d_prime, tc1_prime, tc2_prime, dc1_plus, dc1_minus, tc1_plus, tc1_minus, dc2_plus, dc2_minus, tc2_plus, tc2_minus

# Determine number of exchangeable sites in a peptides sequence
# length of peptide - 2 - #prolines
def ex_sites(seq):
    num_pro = seq.count('P')
    if (seq[0]=='P'):
        num_pro -= 1
    return len(seq)-num_pro-1

# Log-linear interpolation
def log_lin(tx_plus,tx_minus,dx_plus,dx_minus,x):
    q1 = math.log10(tx_plus / tx_minus)
    d1 = dx_plus - dx_minus
    d2 = x - dx_minus
    return 10**(((q1 / d1) * d2) + q1)


def protection_factors():

    keys, constructs, peptides = get_data()

    E3CK_v_E3CKUb = []
    LRRE3CK_v_LRRE3CKUb = []
    LRRE3CK_PKN1_v_LRRE3CKUb_PKN1 = []
    E3CK_v_LRRE3CK = []
    LRRE3CK_v_LRRE3CK_PKN1 = []

    for key in range(1,len(keys)):
        pep_name = keys[key]
        print(key,pep_name)

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

        E3CK_v_E3CKUb.append(calc_pf(E3CK_val, time, E3CKUb_val, time))
        LRRE3CK_v_LRRE3CKUb.append(calc_pf(LRRE3CK_val, time, LRRE3CKUb_val, time))
        LRRE3CK_PKN1_v_LRRE3CKUb_PKN1.append(calc_pf(LRRE3CK_PKN1_val, time, LRRE3CKUb_PKN1_val, time))
        E3CK_v_LRRE3CK.append(calc_pf(E3CK_val, time, LRRE3CK_val, time))
        LRRE3CK_v_LRRE3CK_PKN1.append(calc_pf(LRRE3CK_val, time, LRRE3CK_PKN1_val, time)) 

protection_factors()
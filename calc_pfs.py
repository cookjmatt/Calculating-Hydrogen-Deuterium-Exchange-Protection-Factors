from os import sys
import math
import numpy as np
import pandas as pd
from scipy import stats
from scipy import interpolate

### Get the real residue numbers of a given peptide within the SspH1 sequence
###  Since peptide coverage is based on the E3CK construct MSMS data, that is the sequence used here
def E3CK_get_bounds(s):
    seq='GSHMASIRIHFDMAGPSVPREARALHLAVADWLTSAREGEAAQADRWQAFGLEDNAAAFSLVLDRLRETENFKKDAGFKAQISSWLTQLAEDAALRAKTFAMATEATSTCEDRVTHALHQMNNVQLVHNAEKGEYDNNLQGLVSTGREMFRLATLEQIAREKAGTLALVDDVEVYLAFQNKLKESLELTSVTSEMRFFDVSGVTVSDLQAAELQVKTAENSGFSKWILQWGPLHSVLERKVPERFNALREKQISDYEDTYRKLYDEVLKSSGLVDDTDAERTIGVSAMDSAKKEFLDGLRALVDEVLGSYLTARWRLN'
    l=seq.find(s)+383
    h=l+len(s)-1
    return (np.array([l,h]))
###

### Calculate protection factor
def calc_pf(t, d1_in, d2_in, seq):

    d1 = np.copy(d1_in)
    d2 = np.copy(d2_in)
    error_return_default = [[0,0],[0,0],[0,0],[0,0],-1.0]

    # Calculate range of uptake values shared between d1 and d2 
    r = min(max(d1),max(d2))-max(min(d1),min(d2))
    if (r < 0):
        print("error with {:}".format(seq))
        return error_return_default

    # If a sampled HDX point has lower fractional deuteration than that preceding point, raise the deuteration point to the value
    #  of the preceding point. This ensures proper sampling points for interpolation. This does not change the raw data, it is only
    #  for picking sampling points (see examples)
    for i in range(0,len(d1)-1):
        if (d1[i+1] < d1[i]):
            d1[i+1] = d1[i]
        if (d2[i+1] < d2[i]):
            d2[i+1] = d2[i]

    # Pick sampling times and perform log-linear interpolation to obtain HDX values
    #  Algorithm from Walters et al.
    d_prime = np.linspace(max(min(d1),min(d2)) + 0.001, min(max(d1),max(d2)) - 0.001,10*r)

    # Log-linear interpolation from scipy, this works better than the algorithm above
    f1 = interpolate.interp1d(d1, np.log10(time))
    f2 = interpolate.interp1d(d2, np.log10(time))
    try:
    	t1_prime = 10**f1(d_prime)
    	t2_prime = 10**f2(d_prime)
    except:
    	return error_return_default

    # Calculate the protection factor
    quotient = t2_prime / t1_prime
    pf = stats.gmean(quotient)
    if (np.isnan(pf)):
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
    PF_LRRE3CKUb_LRRE3CKUb_PKN1

    # Arrays of all of the results of data analyis for the protection factors
    all_E3CK_E3CKUb = []
    all_LRRE3CK_LRRE3CKUb = []
    all_LRRE3CK_PKN1_LRRE3CKUb_PKN1 = []
    all_E3CK_LRRE3CK = []
    all_E3CKUb_LRRE3CKUb = []
    all_LRRE3CK_LRRE3CK_PKN1 = []
    all_LRRE3CKUb_LRRE3CKUb_PKN1

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
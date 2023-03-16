#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 16:18:39 2022

@author: madannehl

When training with multiple input channels (wavelengths) and multiple output 
channels, we want each data record to consist of images of all (input+output) 
channels that were taken at the same time. This means that for a specific 
timestamp the images of each input and output channel must exist.

This script helps answering the question: How much timestamps are available for 
the different combinations of AIA input images while using always all three 
channels Bx,By,Bz together as output.


The script takes a look at all possible AIA combinations with respect to the 
specified years. First the script collects all existing timestamps for which 
all of the 3 HMI output channels (x,y,z) exist. This is done first to reduce 
the search space and speed up the script.

Use the parameter n to specify how many AIA wavelengths you want to combine.

The script converts file names to numbers to speed up the loops. This enables
the script to finish in a reasonable amount of time. Also the script uses 
pickle so the combinations for the same years don't have to be recalculated
among multiple runs.

Before running, make sure that fileListYYYY.txt of the selected years exist.
Those files can be generated as follows (example 2013):
cd /storage/databanks/machinelearning/NASA-Solar-Dynamics-Observatory-Atmospheric-Imaging-Assembly/
cd 2013/
find . -type f > ~/fileList2013.txt

"""

import numpy as np
import pickle
import itertools
#import datetime as dt
#import re
#from timeit import default_timer as timer
from tqdm import tqdm



#############################################################################
### Show how many timestamps exist for the given combinations of channels ###
### Please specify:                                                       ###
n = 3 # type here how many AIA channels you would like to combine
year_selection = [2013,2014] # specify the years you are interested in
#############################################################################



TYP_AIA = [
    '0094',
    '0131',
    '0171',
    '0193',
    '0211',
    '0304',
    '0335',
    '1600',
    '1700',
    '4500'
]

TYP_AIA_INT = [int(x) for x in TYP_AIA]

TYP_HMI = [
    'bx',
    'by',
    'bz'    
]

TYP = TYP_AIA + TYP_HMI



year_selection.sort()
lines = []
for year in year_selection:
    with open('file_list_hpc2/fileList'+str(year)+'.txt') as f:
        lines += f.readlines()


fileNamesAIA = []
fileNamesHMI = []
corresponding_types_aia= []
corresponding_types_hmi = []

for l in lines :
    if "npz" in l:
        tag = ""
        if "AIA" in l:
            tag = "AIA"
            fname = l[l.rindex(tag):-5] # remove .npz file ending and directories
            corresponding_types_aia.append(fname[17:]) # extract type
            fileNamesAIA.append(int(fname[3:].replace("_",""))) # remove AIA tag and underscores
        else:
            tag = "HMI"
            fname = l[l.rindex(tag):-5] # remove .npz file ending
            corresponding_types_hmi.append(fname[17:]) # preserve info bx by bz 
            fname = fname.replace("bx", "00").replace("by","11").replace("bz","22") # encode bxyz as numbers
            fileNamesHMI.append(int(fname[3:].replace("_",""))) # remove HMI tag
            #print(fname, l)

sortIdxAIA = np.argsort(fileNamesAIA)
sortIdxHMI = np.argsort(fileNamesHMI)

file_names_aia = np.array(fileNamesAIA)[sortIdxAIA]
file_names_aia_set = set(file_names_aia)
file_names_hmi = np.array(fileNamesHMI)[sortIdxHMI]
corresponding_types_aia = np.array(corresponding_types_aia)[sortIdxAIA]
corresponding_types_hmi = np.array(corresponding_types_hmi)[sortIdxHMI]

all_existing_aia_timestamps = (file_names_aia / 100).astype(int)
all_existing_hmi_timestamps = (file_names_hmi / 100).astype(int)

hmi_unique_timestamps, hmi_counts = np.unique(all_existing_hmi_timestamps, return_counts=True)

relevant_timestamps = hmi_unique_timestamps[np.flatnonzero(hmi_counts==3)] # timestamps with all 3 hmi components x,y,z


# Which possible AIA combinations exist?
def calc_combos(items):

    comb = []
    #comb_raw = list(itertools.combinations_with_replacement(['A','B','C','D','E','F'],6))
    comb_raw = list(itertools.combinations_with_replacement(items,len(items)))
    #print("len_comb_raw:",len(comb_raw))
    for c in comb_raw:
        comb.append(np.unique(c).tolist()) # reduce e.g. AAB to AB
    #print("lencomb0: ", len(comb))
    comb = np.unique(np.asanyarray(comb,dtype=object)) # eliminate multiple ocurrences of AB
    #print("len_comb:",len(comb)) # 1023 for 10 channels is correct: SUM_i=1_n=10 of ( 10 nCr i)
    return comb

comb = calc_combos(TYP_AIA_INT)

### LOAD DATA - if it was created ###
pickle_file = "combinations_"+year_selection.__str__().replace("[","").replace("]","").replace(", ", "_")+".pickle"

results = {}
relevant_timestamps_x10000 = relevant_timestamps * 10000 # speed up the loop
try:
    results = pickle.load(open(pickle_file, "rb"))
    print ("Loaded data.")
except (OSError, IOError):
    # For each possible combination: for which timestamps every channel of the combo exists?
    # or
    # For each relevant timestamp: what aia combos does it fit?
    for t in tqdm(relevant_timestamps_x10000, disable=False, desc="Inspecting which combination the timestamps fit.", unit="timestamps"): 
        for c in comb:
            if set(t + c).issubset(file_names_aia_set):
                combination = ''.join(str(c)) 
                results[combination] = results.get(combination, 0) + 1
    pickle.dump(results, open(pickle_file, "wb"))
    print ("Created data and dumped data.")



print('For n='+str(n),"channels the following combinations exist (sorted):")
sorted_keys = sorted(results, key=results.get)
maxVal = 0
for k in sorted_keys:
    if len(k.strip('][').split(', '))==n:
        val = results[k]

        print(k+" has",val,"timestamps in common.")
        




#### old stuff, first tries, not efficient enough ####
### total images       
# print("Total images:")
# total = 0
# for t in TYP:
#     c = 0
#     for fname in file_names:
#         if re.match("\S+"+t+".npz" , fname):
#             c+=1
#     total += c
#     print(t,c)
# if total != len(file_names):
#     raise Exception("Something went wrong, check all the labels.")
    
# print("\n", "Images per month:")
# for month in np.arange(1,13):
#     for t in TYP:
#         c = 0
#         for fname in file_names:
#             if re.match(t+"2014"+str(month)+"[0-9]{2}_[0-9]{4}_"+t+".npz" , fname):
#                 c+=1
#         print(t, month, c)

#fileNames = []
# for l in lines :
#     if "npz" in l:
#         tag = ""
#         if "AIA" in l:
#             tag = "AIA"
#         else:
#             tag = "HMI"
#         fileName = l[l.rindex(tag):-5] # remove .npz file ending
#         fileNames.append(fileName[3:]) # remove tag HMI / AIA
#         corresponding_types.append(fileName[17:])

### filter for data
# all_existing_hmi_timestamps_raw = []
# for fname in file_names_hmi: # all hmi data
#     stripped = fname.strip() #[3:] if AIA / HMI tag needs to be cut off
#     #if (stripped[:3] == "AIA") and (stripped[17:21] != "1700"):
#     year_str = stripped[0:4]
#     month_str = stripped[4:6]
#     day_str = stripped[6:8]
#     time_str = stripped[9:13]
#     # Target: yyyymmdd_hhmm
#     timestamp_str = year_str + month_str + day_str+time_str
#     all_existing_hmi_timestamps_raw.append(int(timestamp_str))
# all_existing_hmi_timestamps = np.array(all_existing_hmi_timestamps_raw) 

# # print(len(all_existing_timestamps_raw))
# timestamps_unique, inverse, counts = np.unique(all_existing_timestamps_raw, return_inverse=True, return_counts=True)
# timestamps_unique = timestamps_unique.astype(int)

# idx_relevant_unique_timestamps = []
# for idx, t in enumerate(timestamps_unique):
#     isRelevant = True
#     ts = str(t)
#     for typ_hmi in TYP_HMI:
#         if ts + typ_hmi not in file_names:
#             isRelevant = False
#             break
#     if isRelevant:
#         idx_relevant_unique_timestamps.append(idx)
    
#     if idx % 1000 == 0: print(idx)


# ### LOAD DATA ###
# pickle_file = "combinationResults.pickle"

# results = {}
# try:
#     results = pickle.load(open(pickle_file, "rb"))
#     print ("Loaded data.")
# except (OSError, IOError):
    
    
#     for idx, inv in enumerate(inverse):
#         combination = ''.join(corresponding_types[np.flatnonzero(inv==inverse)]) 
#         results[combination] = results.get(combination, 0) + 1
#         if idx % 50000==0: print(idx)
#     pickle.dump(results, open(pickle_file, "wb"))
#     print ("Created data and dumped data.")

# print('The following combinations exist (count of occurences > 1)')
# sorted_keys = sorted(results, key=results.get)
# maxKeyWithHMIxyz = ""
# maxVal = 0
# for k in sorted_keys:
#     val = results[k]
#     if "bxbybz" in k and val > maxVal:
#         maxVal = val
#         maxKeyWithHMIxyz = k

# print("The winner with full HMIxyz data is: \n",maxKeyWithHMIxyz)

# print("Now creating a list of all relevant timestamps for that combination...")

# hmi_keys_raw = "bxbybz"
# hmi_keys = [(hmi_keys_raw[i:i+2]) for i in range(0, len(hmi_keys_raw), 2)]
# aia_keys_raw = maxKeyWithHMIxyz.replace(hmi_keys_raw, "") # cut off 'bxbybz'
# aia_keys = [(aia_keys_raw[i:i+4]) for i in range(0, len(aia_keys_raw), 4)]
# keys = hmi_keys + aia_keys
# relevant_timestamps = []
# for idx, t in enumerate(timestamps_unique):
#     isRelevant = True
#     for key in keys:
#         if t.replace("_","",2)+"_"+key not in file_names:
#             isRelevant = False
#             break
#     if isRelevant:
#         relevant_timestamps.append(t)
    
#     if idx % 1000 == 0: print(idx)


### not performant enough:
# comb = []
# #comb_raw = list(itertools.combinations_with_replacement(['A','B','C','D','E','F'],6))
# comb_raw = list(itertools.combinations_with_replacement(TYP,len(TYP)))
# print("len_comb_raw:",len(comb_raw))
# for c in comb_raw:
#     comb.append(np.unique(c).tolist()) # reduce e.g. AAB to AB
# print("lencomb0: ", len(comb))
# comb = np.unique(comb) # eliminate multiple ocurrences of AB
# print("len_comb:",len(comb))
#
# results = []
# for idx, c in enumerate(comb):
#     hit = 0
#     for dt in timestamps_unique:
#         does_exist_in_all_types = True
#         for typ in c:
#             name_to_check = dt.replace("_","",2)
#             if "b" in typ:
#                 name_to_check = "HMI" + name_to_check + "_" + typ + ".npz"
#             else: 
#                 name_to_check = "AIA" + name_to_check + "_" + typ +".npz"
            
#             if name_to_check not in file_names:
#                 does_exist_in_all_types = False
#                 break
#         if does_exist_in_all_types:
#             hit += 1
#     results.append(hit)
#     print(idx)
            
            

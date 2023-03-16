#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 16:36:10 2022

@author: madannehl

Export of relevant timestamps for a given channel combination: The script exports
only timestamps for which the specified AIA images and all HMI(x,y,z) images
are available. Please specify which years to consider for building the dataset 
and how many months per year to put aside for testing. The specific months 
for testing are picked randomly but are chosen without a gap inside a year.
For instance Jan+Feb or Feb+Mar or Jan+Feb+Mar but never Jan+Mar solely. 
Additionally you can specify a gap between the test and the training set, so 
that the test set will be more independent from the train set.
The script also allows to specify a cadence for the training set and the test 
set separately.

The script converts file names to numbers to speed up the loops. This enables
the script to finish in a reasonable amount of time.

Before running this script, make sure that fileListYYYY.txt of all the selected
years exists in the folder file_list_hpc2. For the procedure how to create those
files follow the instructions in solari2i/code/file_list_hpc2/info.txt.

However, here a short extract of the most important lines (example year 2013):
cd /storage/databanks/machinelearning/NASA-Solar-Dynamics-Observatory-Atmospheric-Imaging-Assembly/
cd 2013/
find . -type f > ~/fileList2013.txt

"""

import numpy as np
import datetime as dt
from calendar import monthrange



################## Settings: Feel free to edit their values ##################
years = [2010,2011,2012,2013,2014,2015,2016,2017]

months_per_year_test = 2 # how many months per year will be picked for testing

cadence_train = 4*60 # the unit is minutes, zero to use all data available
cadence_test  = 4*60 # the unit is minutes, zero to use all data available

test_train_spacing = 15 # the unit is days, max 28, min 1

chosen_combination = [131,1600] # please sort, smallest first
# possible values: [94,131,171,193,211,304,335,1600,4500] 

out_file_train = "relevant_timestamps/xrelevant_timestamps_train.txt"
out_file_test = "relevant_timestamps/xrelevant_timestamps_test.txt"
##############################################################################



years.sort()

lines = []
for year in years:
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
            fname = l[l.rindex(tag):-5] # remove .npz file ending
            corresponding_types_aia.append(fname[17:])
            fileNamesAIA.append(int(fname[3:].replace("_",""))) # remove AIA tag
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

relevant_timestamps_x10000 = relevant_timestamps * 10000 # this conversion enables us to check for the filenames very fast


## pick randomly which months of which years built the test set
## picking only multiple years that directly follow each other ##
test_months = {}
for year in years:
    # reserve one more month (+1) for the spacing before and ofter the test data
    # we use half of the additional month for spacing before and after test data
    monthNrStart = np.random.randint(1,14 - (months_per_year_test + 1))
    #monthNrEnd = monthNrStart + (months_per_year_test - 1)
    
    if year == 2010 and monthNrStart < 5 :
        print("Warning: No data exists for Jan-Apr 2010, verify if the proposed test set suits your needs.")
    
    months = []
    for i in range(months_per_year_test + 1):
        months.append(monthNrStart + i)
        
    test_months[year] = months


## filter and export timestamps ##
last_timestamp_train = dt.datetime(1800,1,1)
last_timestamp_test = dt.datetime(1800,1,1)
counter_train = 0
counter_test = 0
with open(out_file_train, 'wt') as f_train, open(out_file_test, 'wt') as f_test:
    for idx, t in enumerate(relevant_timestamps_x10000):
        if set(t + chosen_combination).issubset(file_names_aia_set):
            # example to show how this condition works:
            # t = 2010051300000000
            # chosen_combination = [171, 193, 304]
            # ->  t + chosen_combination = array([2010051300000171, 2010051300000193, 2010051300000304], dtype=int64)
            # then we can check if setA.issubset(setB) if the given timestamp t exists for all of the chosen channels
            
            ## lets prepare the timestamp for export ## 
            raw_str = str(int(t/10000))# cut off placeholder for iai tag
            year_str = raw_str[:4]
            month_str = raw_str[4:6]
            day_str = raw_str[6:8]
            hour_str = raw_str[8:10]
            minute_str = raw_str[10:12]
            out_str = year_str+"_"+month_str+"_"+day_str+"_"+hour_str+minute_str+"\n" # target format: YYYY_DD_MM_hhmm
                
            timestamp = dt.datetime(int(year_str), int(month_str), int(day_str), int(hour_str), int(minute_str))
            
            if int(month_str) in test_months[int(year_str)]: # check if it is part of the test set
                
                ## spacing before and after the test data ##
                if int(month_str) == test_months[int(year_str)][0] and int(day_str) <= test_train_spacing: 
                    continue
                elif int(month_str) == test_months[int(year_str)][-1] and int(day_str) > monthrange(int(year_str), int (month_str))[1] - test_train_spacing: 
                    continue
                
                
                ### respect the given cadence and export ###
                if (timestamp - last_timestamp_test).total_seconds() / 60 >= cadence_test:
                    f_test.write(out_str)
                    last_timestamp_test = timestamp
                    counter_test += 1
                    
            else: ## train set ## 
                ## respect the given cadence and export ##
                if (timestamp - last_timestamp_train).total_seconds() / 60 >= cadence_train:
                    f_train.write(out_str)
                    last_timestamp_train = timestamp
                    counter_train += 1
                
        
print("Successfully exported "+str(counter_train)+" timestamps to",out_file_train,"and",str(counter_test),"timestamps to",out_file_test,".")
print("The training set consists of",np.round(counter_train/(counter_test+counter_train)*100,1),"% of the timestamps.")
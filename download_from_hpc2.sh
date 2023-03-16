#!/bin/bash

# author: mdannehl
# 
# This is a bash script that downloads specified parts of the SDO dataset from 
# the hpc server. It works as follows: 
# The first part of the script (the loop) parses a file containing the relevant
# timestamps. For each timestamp the script constructs the correct filenames 
# and paths for each selected channel and appends this to a temporary file. 
# To select the input and output channels that you would like to download, 
# uncomment the corresponding lines inside the loop. After the loop you can 
# find the rsync command to download the files specified in the temporary file.
#
# Please take into account, that the temporary file is appended when rerunning 
# the script. This is intentional because it enables us to easily construct the 
# temporary file over multiple runs for different files of relevant timestamps 
# by changing the specified input file and just commenting out the rsync command.
#
# The last command deletes empty (sub-)directories in the target directory that
# are automatically created by rsync.
#
# Be aware that the input file should only contain timestamps for that all of 
# the selected channels (inside the loop) are existing. E.G. If you select the
# channel 0304 and 1600, make sure that all those channels have data on the 
# specified timestamps of the input file.
#
# !! In order to execute this script, make sure that the appropriate file 
# permissions are set. Use e.g. CHMOD +x to add the execute permission. !!
#

##############################################################################
### Please edit as needed:

### Input file that contains the relevant timestamps
#File="relevant_timestamps/relevant_timestamps_171_193_304_2010-2017_c4h_10m_15sp.txt"
File="relevant_timestamps/relevant_timestamps_short_test.txt"


### where to store the downloaded files
#targetDir="../data/solar-dataset_171_193_2010-2017-c4h/"
targetDir="../data/solar-data/"


### name of temporary file
tempOutFile="downloading_files.txt"
##############################################################################

### read the input file with the relevant timestamps
Lines=$(cat $File)

### handle each timestamp that was found in the input file
for line in $Lines
do

    ### extract the date information for an easier concatination of the file names and paths
	year="${line:0:4}"
	month="${line:5:2}"
	day="${line:8:2}"
	time="${line:11:4}"
	
	### Construct the file names and path and append the temporary file, 
	### Feel free to comment/uncomment the following lines to fit your needs:
#	printf "%s\n" "/$year/AIA_0094/$month/$day/AIA$year$month${day}_${time}_0094.npz" >> "${tempOutFile}"
#	printf "%s\n" "/$year/AIA_0131/$month/$day/AIA$year$month${day}_${time}_0131.npz" >> "${tempOutFile}"
	printf "%s\n" "/$year/AIA_0171/$month/$day/AIA$year$month${day}_${time}_0171.npz" >> "${tempOutFile}"
	printf "%s\n" "/$year/AIA_0193/$month/$day/AIA$year$month${day}_${time}_0193.npz" >> "${tempOutFile}"
#	printf "%s\n" "/$year/AIA_0211/$month/$day/AIA$year$month${day}_${time}_0211.npz" >> "${tempOutFile}"
#	printf "%s\n" "/$year/AIA_0304/$month/$day/AIA$year$month${day}_${time}_0304.npz" >> "${tempOutFile}"
#	printf "%s\n" "/$year/AIA_0335/$month/$day/AIA$year$month${day}_${time}_0335.npz" >> "${tempOutFile}"
#	printf "%s\n" "/$year/AIA_1600/$month/$day/AIA$year$month${day}_${time}_1600.npz" >> "${tempOutFile}"
#	printf "%s\n" "/$year/AIA_1700/$month/$day/AIA$year$month${day}_${time}_1700.npz" >> "${tempOutFile}"
#	printf "%s\n" "/$year/AIA_4500/$month/$day/AIA$year$month${day}_${time}_4500.npz" >> "${tempOutFile}"
#	printf "%s\n" "/$year/HMI_Bx/$month/$day/HMI$year$month${day}_${time}_bx.npz" >> "${tempOutFile}"
#	printf "%s\n" "/$year/HMI_By/$month/$day/HMI$year$month${day}_${time}_by.npz" >> "${tempOutFile}"
#	printf "%s\n" "/$year/HMI_Bz/$month/$day/HMI$year$month${day}_${time}_bz.npz" >> "${tempOutFile}"


done

### the following line starts downloading from the server that is here called "meso"
rsync -rav --info=PROGRESS2,COPY0,SKIP0 --include-from="${tempOutFile}" --exclude='*.*' meso:/storage/databanks/machinelearning/NASA-Solar-Dynamics-Observatory-Atmospheric-Imaging-Assembly/ "${targetDir}" 

### the following
cd "${targetDir}" && find . -type d -empty -delete

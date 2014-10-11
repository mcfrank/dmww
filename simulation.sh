# script for a running a single dmww simulation on the proclus grid engine

# run job using bash from directory it was submitted from
#$ -cwd
#$ -S /bin/bash

# merge error output and standard output
#$ -j y

# rerun job if aborted
#$ -r y

# specify that reservation should be done
#$ -R y

# set validation level to reject invalid request
#$ -w e

# email notifications for beginning/end/aborted/suspended to address
#$ -M mikabr@stanford.edu
#$ -m beas

# set a name
#$ -N $1"-"$2"-"$3"-"$4"-"$5"-"$6

echo "Preparing to run simulation with parameters "$*

echo "Loading python modules..."
module load python

echo "Starting simulation..."
python dmww_testing.py -a $1 -n $2 -b $3 --alpha-r $4 --alpha-nr $5 --empty-intent $6

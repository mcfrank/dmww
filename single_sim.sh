#!/bin/bash
# run a single dmww simulation using given algorithm and number of samples/particles
# run with ./grid_sim.sh <algorithm> <number of samples/particles>

qsub simulation.sh $1 $2 $3 $4 $5

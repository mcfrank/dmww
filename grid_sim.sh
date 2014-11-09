#!/bin/bash
# run a grid of dmww simulations using given algorithm and number of samples/particles
# run with ./grid_sim.sh <algorithm> <number of samples/particles> <number of chains>

ar_opts="10 1.0 0.1 0.01"
an_opts="100 10 1.0 0.1"
ei_opts="0.1 0.01 0.001"

for c in $(seq 1 $3)
do
  for ar in $ar_opts
  do
    for an in $an_opts
    do
      for ei in $ei_opts
      do
        qsub simulation.sh $1 $2 $ar $an $ei
      done
    done
  done
done

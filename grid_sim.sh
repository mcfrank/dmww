#!/bin/bash
# run a grid of dmww simulations using given algorithm and number of samples/particles
# run with ./grid_sim.sh <algorithm> <number of samples/particles> <burn-in period>

ar_opts="1.0 0.1 0.01 0.001 0.0001"
an_opts="100 10 1.0 0.1 0.01"
ei_opts="0.1 0.01 0.001"

for ar in $ar_opts
do
  for an in $an_opts
  do
    for ei in $ei_opts
    do
      qstat simulation.sh $1 $2 $3 $ar $an $ei
    done
  done
done

#!/bin/bash
# run a grid of dmww particle simulations using given number of particles
# run with ./pf_type_sim.sh <number of samples/particles> <corpus type>

ar_opts="1.0"
an_opts="1.0"
ei_opts="0.001"

#for c in $(seq 1 $3)
#do
  for ar in $ar_opts
  do
    for an in $an_opts
    do
      for ei in $ei_opts
      do
        qsub simulation.sh pf $1 $2 $ar $an $ei
      done
    done
  done
#done

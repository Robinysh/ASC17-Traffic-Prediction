#!/bin/bash
#PBS -S /bin/bash
#PBS -N TrafficGA
#PBS -q normal
#PBS -l nodes=8:ppn=24
#PBS -l walltime=24:00:00
#PBS -o TrafficGA.out
#PBS -e TrafficGA.err
cd /home/ppc17public/Q3/Code/asc-17-traffic-prediction-no-txt-output/

/home/ppc17public/Q3/Code/asc-17-traffic-prediction-no-txt-output/tensorflow.sh /home/ppc17public/Q3/Code/asc-17-traffic-prediction-no-txt-output/helloevolve.py

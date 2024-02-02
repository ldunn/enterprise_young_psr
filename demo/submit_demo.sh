#!/bin/bash
#SBATCH --job-name=enterprise
#SBATCH --output=logs/enterprise_out_%A.dat

#SBATCH --ntasks=32
#SBATCH --mem-per-cpu=2G
#SBATCH --time=24:00:00
#SBATCH --tmp=2G

### REPLACE THESE LINES WITH LINES APPROPRIATE FOR YOUR ENVIRONMENT ###
source ~/activate_nt.sh
workon gc_search_nt
######

cp -r $TEMPO2 $JOBFS/tempo2
cp -r $TEMPO2/clock $JOBFS/tempo2_clock_dir
export TEMPO2=$JOBFS/tempo2
export TEMPO2_CLOCK_DIR=$JOBFS/tempo2_clock_dir

for MODEL in TN TNLONG; do

    mpirun stdbuf -e0 -o0 python ../run_psr_multinest.py --par J0835-4510_glitch1.par --tim J0835-4510_glitch1.tim --model $MODEL --out-dir results_multinest/$MODEL --nlive 400 --timing-param-range 5000 --log-params "GLF0,GLTD" --equad-range " -6, -3"
    
    python ../make_corner_plot_multinest.py --psrj J0835-4510_glitch1 --in-par J0835-4510_glitch1.par --in-tim J0835-4510_glitch1.tim --model $MODEL --plot-dir results_multinest/$MODEL --results-dir results_multinest/$MODEL --log-params "GLF0,GLTD"
    python ../get_enterprise_par_multinest.py --psrj J0835-4510_glitch1 --in-par J0835-4510_glitch1.par --in-tim J0835-4510_glitch1.tim --model $MODEL --par-out-dir results_multinest/$MODEL --results-dir results_multinest/$MODEL --output-tn-params --plot-dir results_multinest/$MODEL/ --log-params "GLF0,GLTD"
done

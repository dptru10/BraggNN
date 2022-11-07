#!/bin/bash


##SBATCH --ntasks=1
##SBATCH --cpus-per-task=28
##SBATCH --nodes=1
##SBATCH --partition=mercurial
##SBATCH -J braggNN 
###SBATCH --time=12:00:00
##SBATCH -e ./%J.err
##SBATCH -o ./%J.out

# execute program

h5dirSim=/home/dtrujillo/PeakGenerator/h5_out
SimPeakLoc=peaks-exp4train-simulated-psz15-npeaks-5.0e+05.hdf5  
SimFrames=frames_5.0e+05.h5 

h5dirExp=/home/dtrujillo/experimental_data_feb21_402_410
ExpPeakLoc=peaks-exp4train-psz15-npeaks-5.0e+05.hdf5  
ExpFrames=frames_exp_5.0e+05.h5 

h5demodir=/home/dtrujillo/BraggNN/dataset
DemoPeakLoc=peaks-exp4train-simulated-psz15-npeaks-5.0e+05.hdf5  
DemoFrames=frames_5.0e+05.h5 

h5midasTest=/home/dtrujillo/feb21_individual_h5_files
MidasPeakLoc=PeakInfo-feb21-000402.h5

geDir=/home/dtrujillo/feb21_geFiles
geFrameFile=park_sam_cx5b_s0_ff_with_Pb_000402.ge3
geDarkFile=dark_before_000384.ge3

#only need to change these 2 lines
h5dir=$h5dirExp
locs=$ExpPeakLoc
frames=$ExpFrames


echo "program started"
echo "Using $h5midasTest in $MidasPeakLoc in inference mode"  
#date

##500K simulated data block
#python3 -u main-fixed.py -fcsz='32_16_8_4' -p_file $h5dir/peaks-exp4train-simulated-psz15-npeaks-5.0e+05.hdf5 -f_file $h5dir/frames_5.0e+05.h5 -expName simulated_5e5_fcsz_32_16_8_4 -maxep 10
#python3 -u inference.py  -m_file=simulated_5e5-itrOut/mdl_simulated_5e5.pth -p_file $h5dir/peaks-exp4train-simulated-psz15-npeaks-5.0e+05.hdf5 -f_file $h5dir/frames_5.0e+05.h5 -expName sim_mdl_sim_5e5 

##500K experimental data block
#python3 -u main-fixed.py  -p_file $h5dirExp/peaks-exp4train-psz15-npeaks-5.0e+05.hdf5 -f_file $h5dirExp/frames_exp_5.0e+05.h5 -expName exp_feb21_402_410_npeaks_5e5 -maxep 10
#python3 -u inference.py  -m_file=exp_feb21_402_410_npeaks_5e5-itrOut/mdl_exp_feb21_402_410_npeaks_5e5.pth -p_file $h5dir/peaks-exp4train-psz15-npeaks-5.0e+05.hdf5 -f_file $h5dir/frames_exp_5.0e+05.h5 -expName exp_mdl_exp_data_5e5 

##500K simulated data block
#best set of hyperparamters:{'lr': 0.0007341007811574974, 'batch_size': 64} see 42.out 
#python3 -u main-fixed.py  -p_file $h5dirSim/peaks-exp4train-simulated-psz15-npeaks-5.0e+05.hdf5 -f_file $h5dirSim/frames_5.0e+05.h5 -expName simulated_5e5 -maxep 10
#python3 -u inference.py  -m_file=simulated_5e5-itrOut/mdl_simulated_5e5.pth -p_file $h5dirSim/peaks-exp4train-simulated-psz15-npeaks-5.0e+05.hdf5 -f_file $h5dirSim/frames_5.0e+05.h5 -expName sim_mdl_sim_non_raytune_5e5 

##10K simulated data block
#python3 -u main-fixed.py  -p_file $h5dir/peaks-exp4train-simulated-psz15-npeaks-1.0e+04.hdf5 -f_file $h5dir/frames_1.0e+04.h5 -expName simulated_1e4 -maxep 10
#python3 -u inference.py  -m_file=simulated_1e4-itrOut/mdl_simulated_1e4.pth -p_file $h5dir/peaks-exp4train-simulated-psz15-npeaks-1.0e+04.hdf5 -f_file $h5dir/frames_1.0e+04.h5 -expName sim_mdl_sim_1e4 

##sim-sim-mix 
#python3 -u inference.py  -m_file=simulated_5e5-itrOut/mdl_simulated_5e5.pth -p_file $h5dir/peaks-exp4train-psz15-npeaks-5.0e+05.hdf5 -f_file $h5dir/frames_exp_5.0e+05.h5 -expName simulated_mdl_exp_data_5e5 

##sim-exp-mix 
#python3 -u inference.py  -m_file=simulated_5e5_raytune-itrOut/mdl_simulated_5e5_raytune.pth -p_file $h5dirExp/peaks-exp4train-psz15-npeaks-5.0e+05.hdf5 -f_file $h5dirExp/frames_exp_5.0e+05.h5 -expName simulated_mdl_exp_data_5e5 

##existing-demo-data-mix
#best set of hyperparamters:{'lr': 0.0015586280395314678, 'batch_size': 128} see 42.out 
#python3 -u main-raytune.py  -p_file $h5demodir/peaks-exp4train-psz15.hdf5 -f_file $h5demodir/frames.h5 -expName sim_mdl_demo_non_raytune_5e5 -maxep 1000

##exp-sim-mix 
#python3 -u inference.py  -m_file=exp_feb21_402_410_npeaks_5e5-itrOut/mdl_exp_feb21_402_410_npeaks_5e5.pth -p_file $h5dirSim/peaks-exp4train-simulated-psz15-npeaks-5.0e+05.hdf5 -f_file $h5dirSim/frames_5.0e+05.h5 -expName exp_mdl_sim_data_5e5 



##RayTune block 
##500K simulated data block
#python3 -u main-raytune.py  -p_file $h5dir/$locs -f_file $h5dir/$frames -expName simulated_5e5_raytune -maxep 20
#python3 -u main-raytune.py  -m_file $h5midasTest/$MidasPeakLoc -expName individual_midas_input_402_raytune -maxep 20
#python3 -u main-fixed.py  -lr 0.0007341007811574974 -mbsz 64  -m_file $h5midasTest/$MidasPeakLoc -expName individual_midas_input_402_raytune -maxep 20
#python3 -u main-fixed.py -lr 0.0007341007811574974 -mbsz 64  -p_file $h5dir/$locs -f_file $h5dir/$frames -expName simulated_5e5_raytune -maxep 10
#python3 -u main-fixed.py  -p_file $h5dir/$locs -f_file $h5dir/$frames -expName simulated_5e5_raytune -maxep 25
#python3 -u inference.py  -m_file=simulated_5e5_raytune-itrOut/mdl_simulated_5e5_raytune.pth -p_file $h5dir/$locs -f_file $h5dir/$frames -expName sim_mdl_sim_raytune_5e5 
#python3 -u inference.py  -midas_file $h5midasTest/$MidasPeakLoc -m_file=simulated_5e5-itrOut/mdl_simulated_5e5.pth -expName sim_mdl_midas_feb21_402 

#ge3_test
#python3 -u main-fixed.py  -lr 0.0007341007811574974 -mbsz 64  -ge_ffile $geDir/$geFrameFile -ge_dfile $geDir/$geDarkFile -p_file $h5dir/$locs -expName individual_midas_input_402_raytune -maxep 20
python3 -u inference.py  -ge_ffile $geDir/$geFrameFile -ge_dfile $geDir/$geDarkFile -m_file=simulated_5e5-itrOut/mdl_simulated_5e5.pth -expName sim_mdl_midas_feb21_402 

echo "program done"
#date

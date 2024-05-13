#!/bin/bash
# Run different command according to different program_type.
# An example like this:
#   bash script_oneDGaussian.sh --program_type oneD_Gaussian_DAEBM
# program_type:
#   oneD_Gaussian
#   oneD_Gaussian_DAEBM
#   oneD_Gaussian_DAEBM_v1
#   post_sampling_oneD
#   testforenergy

# Parse args
while [[ $# -gt 0 ]];do
    case "$1" in
        --program_type)
            program_type=$2
            shift 2
            ;;
        
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

case $program_type in
    "oneD_Gaussian_DAEBM")

        python ./oneD_Gaussian_DAEBM.py  --num_of_points 1000 --means -3 -1 1 3 --peaks_ratio 1 3 3 1 --t_seed 2024 \
                        --print_freq 50 --cuda 1 --log_dir runs/onedimgauss_exp \
                        --betas 0.0 0.999 --optimizer_type adam --lr 5e-3 --n_epochs 200 --time_embedding_type sin \
                        --scheduler_type MultiStep --milestones  160 180 220 --lr_decay_factor 0.1 \
                        --batch_size 100 --n_warm_epochs 10  --act_func Softplus \
                        --sample_steps 50 --replay_buffer_size 1000 --b_factor 3e-1 --random_type normal \
                        --diffusion_betas 1e-2 2e-1  --num_diffusion_timesteps 6 --diffusion_schedule sqrtcumlinear \
                        --start_reject_epochs 1 --dynamic_sampling 
        ;;
    "oneD_Gaussian_DAEBM_v1")
        python ./oneD_Gaussian_DAEBM_v1.py  --num_of_points 1000 --t_seed 2024 --print_freq 10 --cuda 6 --log_dir runs/onedimgauss_v1_exp \
                        --betas 0.0 0.999 --optimizer_type adam --lr 5e-3 --n_epochs 200 \
                        --scheduler_type MultiStep --milestones 80 120 160 --lr_decay_factor 0.1 \
                        --batch_size 100 --n_warm_epochs 10  --act_func Softplus \
                        --sample_steps 50 --replay_buffer_size 1000 --b_factor 3e-1 \
                        --diffusion_betas 1e-2 2e-1  --num_diffusion_timesteps 6 --diffusion_schedule sqrtcumlinear \
                        --start_reject_epochs 1 --dynamic_sampling 
        ;;
    "post_sampling_oneD")
        python ./post_sampling_oneD.py --t_seed 2024 --num_of_points 1000  --stop_a_chain_M 50 \
                                        --total_iters 1000 --sampling_chains 1000 --num_diffusion_timesteps 10 \
                                        --diffusion_schedule sqrtcumlinear --diffusion_betas 1e-2 0.15 \
                                        --mala_isreject True --sample_steps 50 --dynamic_sampling
        ;;
    "testforenergy")
        python ./testforenergy.py --t_seed 2024 --num_of_points 1000  --stop_a_chain_M 50 --batch_size 100 --log_dir runs/net_test \
                                        --total_iters 1000 --sampling_chains 1000 --num_diffusion_timesteps 6 \
                                        --diffusion_schedule sqrtcumlinear --diffusion_betas 1e-2 2e-1 \
                                        --mala_isreject True --sample_steps 50 --dynamic_sampling
        ;;
    "oneD_Gaussian")
        python ./oneD_Gaussian.py --log_dir runs/onedimebm_exp --num_of_points 1000 --mala_sigma 0.1 --mala_steps 40 \
                            --act_func Softplus --batch_size 100 --n_iter 1200 --lr 2e-1 --weight_decay 0 \
                            --milestones 60 80 100 --lr_decay_factor 2e-1 --mala_isreject
        ;;
    *)
        echo "Invalid program type: $program_type"
        exit 1
        ;;
esac
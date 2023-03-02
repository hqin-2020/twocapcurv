#! /bin/bash

# Deltaarray=(0.005 0.001 0.01 0.1 1)
Deltaarray=(300)
fractionarray=(0.0)

actiontime=1

julia_name="newsets_twocapitals_rho.jl"
python_name="plot_rho_3d.py"

rhoarray=(0.7 0.8 0.9 1.00001 1.1 1.2 1.3 1.4 1.5)
# rhoarray=(1.00001)

gammaarray=(1.01 2.0 3.0 4.0 5.0 6.0 7.0 8.0)

symmetric=1
alpha_z_tilde_ex=-0.005

for Delta in ${Deltaarray[@]}; do
    for fraction in "${fractionarray[@]}"; do
        for rho in "${rhoarray[@]}"; do
            for gamma in "${gammaarray[@]}"; do
                    count=0

                    action_name="Standard_grid_sym_xi_nonzero_Delta_300"

                    dataname="${action_name}_${Delta}_frac_${fraction}"

                    mkdir -p ./job-outs/${action_name}/Delta_${Delta}_frac_${fraction}/

                    if [ -f ./bash/${action_name}/Delta_${Delta}_frac_${fraction}/rho_${rho}_gamma_${gamma}.sh ]; then
                        rm ./bash/${action_name}/Delta_${Delta}_frac_${fraction}/rho_${rho}_gamma_${gamma}.sh
                    fi

                    mkdir -p ./bash/${action_name}/Delta_${Delta}_frac_${fraction}/

                    touch ./bash/${action_name}/Delta_${Delta}_frac_${fraction}/rho_${rho}_gamma_${gamma}.sh

                    tee -a ./bash/${action_name}/Delta_${Delta}_frac_${fraction}/rho_${rho}_gamma_${gamma}.sh <<EOF
#!/bin/bash

#SBATCH --account=pi-lhansen
#SBATCH --job-name=${Delta}_${gamma}
#SBATCH --output=./job-outs/$job_name/${action_name}/Delta_${Delta}_frac_${fraction}/rho_${rho}_gamma_${gamma}.out
#SBATCH --error=./job-outs/$job_name/${action_name}/Delta_${Delta}_frac_${fraction}/rho_${rho}_gamma_${gamma}.err
#SBATCH --time=0-12:00:00
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

module load julia/1.7.3
module load python/anaconda-2020.11
srun julia /project/lhansen/twocaprho/$julia_name  --Delta ${Delta} --fraction ${fraction} --gamma ${gamma} --rho ${rho} --symmetric ${symmetric} --alpha_z_tilde_ex ${alpha_z_tilde_ex} --dataname ${dataname} 
python3 /project/lhansen/twocaprho/$python_name  --Delta ${Delta} --fraction ${fraction} --gamma ${gamma} --rho ${rho} --symmetric ${symmetric} --dataname ${dataname}
EOF
                count=$(($count + 1))
                sbatch ./bash/${action_name}/Delta_${Delta}_frac_${fraction}/rho_${rho}_gamma_${gamma}.sh
            done
        done
    done
done
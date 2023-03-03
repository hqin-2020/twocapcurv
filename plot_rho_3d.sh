#! /bin/bash

Deltaarray=(300)
fractionarray=(0.0)

actiontime=1

python_name="plot_rho_3d.py"

rhoarray=(0.7 0.8 0.9 1.00001 1.1 1.2 1.3 1.4 1.5)
rhoarray=(1.0)

gammaarray=(8.0)
gammaarray=(1.01 2.0 3.0 4.0 5.0 6.0 7.0 8.0)

symmetric=0

for Delta in ${Deltaarray[@]}; do
    for fraction in "${fractionarray[@]}"; do
        for rho in "${rhoarray[@]}"; do
            for gamma in "${gammaarray[@]}"; do
                    count=0

                    action_name="Standard_grid_asym_Delta_test"

                    dataname="${action_name}_${Delta}_frac_${fraction}"

                    mkdir -p ./job-outs/${action_name}/Delta_${Delta}_frac_${fraction}/

                    if [ -f ./bash/${action_name}/Delta_${Delta}_frac_${fraction}/rho_${rho}_gamma_${gamma}_plot.sh ]; then
                        rm ./bash/${action_name}/Delta_${Delta}_frac_${fraction}/rho_${rho}_gamma_${gamma}_plot.sh
                    fi

                    mkdir -p ./bash/${action_name}/Delta_${Delta}_frac_${fraction}/

                    touch ./bash/${action_name}/Delta_${Delta}_frac_${fraction}/rho_${rho}_gamma_${gamma}_plot.sh

                    tee -a ./bash/${action_name}/Delta_${Delta}_frac_${fraction}/rho_${rho}_gamma_${gamma}_plot.sh <<EOF
#!/bin/bash

#SBATCH --account=pi-lhansen
#SBATCH --job-name=p_${Delta}_${fraction}
#SBATCH --output=./job-outs/$job_name/${action_name}/Delta_${Delta}_frac_${fraction}/rho_${rho}_gamma_${gamma}_plot.out
#SBATCH --error=./job-outs/$job_name/${action_name}/Delta_${Delta}_frac_${fraction}/rho_${rho}_gamma_${gamma}_plot.err
#SBATCH --time=0-12:00:00
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G

module load python/anaconda-2020.11
python3 /project/lhansen/twocapcurv/$python_name  --Delta ${Delta} --fraction ${fraction} --gamma ${gamma} --rho ${rho} --symmetric ${symmetric} --dataname ${dataname}
EOF
                count=$(($count + 1))
                sbatch ./bash/${action_name}/Delta_${Delta}_frac_${fraction}/rho_${rho}_gamma_${gamma}_plot.sh
            done
        done
    done
done
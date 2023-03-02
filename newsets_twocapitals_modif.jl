#=============================================================================#
#  Economy with TWO CAPITAL STOCKS
#
#  Author: Balint Szoke
#  Date: Sep 2018
#=============================================================================#

using Pkg
using Optim
using Roots
using NPZ
using Distributed
using CSV
using Tables
using ArgParse

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--gamma"
            help = "gamma"
            arg_type = Float64
            default = 8.0
        "--rho"
            help = "rho"
            arg_type = Float64
            default = 1.00001    
        "--fraction"
            help = "fraction"
            arg_type = Float64
            default = 0.01   
        "--Delta"
            help = "Delta"
            arg_type = Float64
            default = 1000.   
        "--dataname"
            help = "dataname"
            arg_type = String
            default = "output"
    end
    return parse_args(s)
end

#==============================================================================#
# SPECIFICATION:
#==============================================================================#
@show parsed_args = parse_commandline()
gamma                = parsed_args["gamma"]
rho                  = parsed_args["rho"]
fraction             = parsed_args["fraction"]
Delta                = parsed_args["Delta"]
dataname             = parsed_args["dataname"]

symmetric_returns    = 1
state_dependent_xi   = 0
optimize_over_ell    = 0
compute_irfs         = 0                    # need to start julia with "-p 5"

if compute_irfs == 1
    @everywhere include("newsets_utils_modif.jl")
elseif compute_irfs ==0
    include("newsets_utils_modif.jl")
end

println("=============================================================")
if symmetric_returns == 1
    println(" Economy with two capital stocks: SYMMETRIC RETURNS          ")
    if state_dependent_xi == 0
        println(" No tilting (xi is NOT state dependent)                      ")
        filename = (compute_irfs==0) ? "model_sym_HS.npz" : "model_sym_HS_p.npz";
    elseif state_dependent_xi == 1
        println(" With tilting (change in kappa)                        ")
        filename = (compute_irfs==0) ? "model_sym_HSHS.npz" : "model_sym_HSHS_p.npz";
    elseif state_dependent_xi == 2
        println(" With tilting (change in beta)                        ")
        filename = (compute_irfs==0) ? "model_sym_HSHS2.npz" : "model_sym_HSHS2_p.npz";
    end
elseif symmetric_returns == 0
    println(" Economy with two capital stocks: ASYMMETRIC RETURNS         ")
    if state_dependent_xi == 0
        println(" No tilting (xi is NOT state dependent)                      ")
        filename = (compute_irfs==0) ? "model_asym_HS.npz" : "model_asym_HS_p.npz";
    elseif state_dependent_xi == 1
        println(" With tilting (change in kappa)                        ")
        filename = (compute_irfs==0) ? "model_asym_HSHS.npz" : "model_asym_HSHS_p.npz";
    elseif state_dependent_xi == 2
        println(" With tilting (change in beta)                        ")
        filename = (compute_irfs==0) ? "model_asym_HSHS2.npz" : "model_asym_HSHS2_p.npz";
    end
end


filename_ell = "./output/"*dataname*"/"

#==============================================================================#
#  PARAMETERS
#==============================================================================#

# (1) Baseline model
a11 = 0.014
alpha = 0.05
zeta = 0.5
kappa = 0.0

scale = 1.32
sigma_k1 = scale*[.0048,               .0,   .0];
sigma_k2 = scale*[.0              , .0048,   .0];
sigma_z =  [.011*sqrt(.5)   , .011*sqrt(.5)   , .025];

eta1 = 0.013
eta2 = 0.013
beta1 = 0.01
beta2 = 0.01

delta = .002;

phi1 = 28.0
phi2 = 28.0

# (3) GRID
II, JJ = 1001, 201;
rmax =  log(20);
rmin = -log(20); 
zmax = 1.;
zmin = -zmax;

# (4) Iteration parameters
maxit = 500000;        # maximum number of iterations in the HJB loop
crit  = 10e-6;      # criterion HJB loop
# Delta = 1000.;      # delta in HJB algorithm


# Initialize model objects -----------------------------------------------------
baseline1 = Baseline(a11, zeta, kappa, sigma_z, beta1, eta1, sigma_k1, delta);
baseline2 = Baseline(a11, zeta, kappa, sigma_z, beta2, eta2, sigma_k2, delta);
technology1 = Technology(alpha, phi1);
technology2 = Technology(alpha, phi2);
model = TwoCapitalEconomy(baseline1, baseline2, technology1, technology2);

grid = Grid_rz(rmin, rmax, II, zmin, zmax, JJ);
params = FinDiffMethod(maxit, crit, Delta);

#==============================================================================#
# WITH ROBUSTNESS
#==============================================================================#

println(" (3) Compute value function WITH ROBUSTNESS")
A, V, val, d1_F, d2_F, d1_B, d2_B, h1_F, h2_F, hz_F, h1_B, h2_B, hz_B,
        mu_1_F, mu_1_B, mu_r_F, mu_r_B, mu_z, V0, rr, zz, pii, dr, dz =
        value_function_twocapitals(gamma, rho, fraction, model, grid, params, symmetric_returns);
println("=============================================================")

g_dist, g = stationary_distribution(A, grid)

# Define Policies object
policies  = PolicyFunctions(d1_F, d2_F, d1_B, d2_B,
                            -h1_F/ell_star, -h2_F/ell_star, -hz_F/ell_star,
                            -h1_B/ell_star, -h2_B/ell_star, -hz_B/ell_star);

# Construct drift terms under the baseline
mu_1 = (mu_1_F + mu_1_B)/2.;
mu_r = (mu_r_F + mu_r_B)/2.;
# ... under the worst-case model
h1_dist = (policies.h1_F + policies.h1_B)/2.;
h2_dist = (policies.h2_F + policies.h2_B)/2.;
hz_dist = (policies.hz_F + policies.hz_B)/2.;

######
d1 = (policies.d1_F + policies.d1_B)/2;
d2 = (policies.d2_F + policies.d2_B)/2;
h1, h2, hz = -h1_dist, -h2_dist, -hz_dist;

CSV.write(filename_ell*"d1.csv",  Tables.table(d1), writeheader=false)
CSV.write(filename_ell*"d2.csv",  Tables.table(d2), writeheader=false)
CSV.write(filename_ell*"h1.csv",  Tables.table(h1), writeheader=false)
CSV.write(filename_ell*"h2.csv",  Tables.table(h2), writeheader=false)
CSV.write(filename_ell*"hz.csv",  Tables.table(hz), writeheader=false)

results = Dict("delta" => delta,
# Two capital stocks
"eta1" => eta1, "eta2" => eta2, "a11"=> a11, 
"beta1" => beta1, "beta2" => beta2,
"sigma_k1" => sigma_k1, "sigma_k2" => sigma_k2,
"sigma_z" =>  sigma_z, "alpha" => alpha, "kappa" => kappa,"zeta" => zeta, "phi1" => phi1, "phi2" => phi2,
"I" => II, "J" => JJ,
"rmax" => rmax, "rmin" => rmin, "zmax" => zmax, "zmin" => zmin,
"rr" => rr, "zz" => zz, "pii" => pii, "dr" => dr, "dz" => dz, "T" => hor,
"maxit" => maxit, "crit" => crit, "Delta" => Delta, "inner" => inner,
"g_dist" => g_dist, "g" => g,
# Robust control under baseline
"V0" => V0, "V" => V, "val" => val, "gamma" => gamma,"rho" => rho,
"d1_F" => d1_F, "d2_F" => d2_F,
"d1_B" => d1_B, "d2_B" => d2_B,
"d1" => d1, "d2" => d2,
"h1_F" => policies.h1_F, "h2_F" => policies.h2_F, "hz_F" => policies.hz_F,
"h1_B" => policies.h1_B, "h2_B" => policies.h2_B, "hz_B" => policies.hz_B,
"h1_dist" => h1_dist, "h2_dist" => h2_dist, "hz_dist" => hz_dist,
"h1" => h1, "h2" => h2, "hz" => hz,
"mu_1" => mu_1, "mu_r" => mu_r, "mu_z" => mu_z)

npzwrite(filename_ell*filename, results)

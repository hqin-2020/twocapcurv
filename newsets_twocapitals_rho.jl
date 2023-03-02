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
        "--symmetric"
            help = "symmetric"
            arg_type = Int
            default = 0
        "--alpha_z_tilde_ex"
            help = "alpha_z_tilde_ex"
            arg_type = Float64
            default = -.005   
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
symmetric            = parsed_args["symmetric"]
alpha_z_tilde_ex     = parsed_args["alpha_z_tilde_ex"]
dataname             = parsed_args["dataname"]
ell_ex = 1/(gamma-1)

symmetric_returns    = symmetric
state_dependent_xi   = 0
optimize_over_ell    = 0
compute_irfs         = 0                    # need to start julia with "-p 5"

if compute_irfs == 1
    @everywhere include("newsets_utils_rho.jl")
elseif compute_irfs ==0
    include("newsets_utils_rho.jl")
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

filename_ell = "./output/"*dataname*"/gamma_"*string(gamma)*"_rho_"*string(rho)*"/"
isdir(filename_ell) || mkpath(filename_ell)

#==============================================================================#
#  PARAMETERS
#==============================================================================#
delta = .002;

# (0) Single capital economy
alpha_c_hat = .484;
beta_hat = 1.0;
sigma_c = [.477, .0];

#===========================  CALIBRATION  ====================================#
# consumption_investment = 3.1;
#A_1cap, phi_1cap, alpha_k_hat, investment_capital = calibration2(15.,
#                                             consumption_investment,
#                                             alpha_c_hat, delta, sigma_c)
# A_1cap, phi_1cap, alpha_k_hat = calibration3(investment_capital,
#                                   consumption_investment,
#                                   alpha_c_hat, delta, sigma_c)
#

A_1cap = .05
phi_1cap = 28.
investment_capital, consumption_investment, alpha_k_hat = calibration3(phi_1cap,
                                            A_1cap, delta, alpha_c_hat, sigma_c)

println("  Calibrated values: A:", A_1cap,
        "  phi_1cap: ", phi_1cap,
        "  alpha_k : ", alpha_k_hat,
        "  C/I : ", consumption_investment,
        "  I/K : ", investment_capital)
println("=============================================================")
#==============================================================================#

# (1) Baseline model
alpha_z_hat = .0;
kappa_hat = .014;
zbar = alpha_z_hat/kappa_hat;
sigma_z_1cap = [.011, .025];

sigma_z =  [.011*sqrt(.5)   , .011*sqrt(.5)   , .025];


if symmetric_returns == 1

    beta2_hat = beta1_hat = 0.5;

    # (2) Technology
    phi2 = phi1 = phi_1cap;
    A2 = A1 = A_1cap;

    if state_dependent_xi == 0
        # Constant tilting function
        scale = 1.754;
        # scale = 1.32;
        alpha_k2_hat = alpha_k1_hat = alpha_k_hat;

        # Worrisome model
        alpha_z_tilde  = alpha_z_tilde_ex;
        kappa_tilde    = kappa_hat;
        alpha_k1_tilde = alpha_k1_hat
        beta1_tilde    = beta1_hat
        alpha_k2_tilde = alpha_k2_hat
        beta2_tilde    = beta2_hat

        ell_star = ell_ex#0.055594409575544096

    elseif state_dependent_xi == 1
        # State-dependent tilting function (fixed kappa, alpha targets q)
        scale = 1.62
        alpha_k2_hat = alpha_k1_hat = alpha_k_hat;

        alpha_z_tilde  = alpha_z_tilde_ex;
        kappa_tilde    =  .005
        alpha_k1_tilde = alpha_k1_hat
        beta1_tilde    = beta1_hat
        alpha_k2_tilde = alpha_k2_hat
        beta2_tilde    = beta2_hat

        ell_star = ell_ex#0.13852940062708508

    elseif state_dependent_xi == 2
        # State-dependent tilting function
        scale = 1.568
        alpha_k2_hat = alpha_k1_hat = alpha_k_hat;

        alpha_z_tilde  = alpha_z_tilde_ex;
        kappa_tilde    = kappa_hat
        alpha_k1_tilde = alpha_k1_hat
        beta1_tilde    = beta1_hat + .1941
        alpha_k2_tilde = alpha_k2_hat
        beta2_tilde    = beta2_hat + .1941

        ell_star = ell_ex#0.18756641482672026

    end


elseif symmetric_returns == 0

    beta1_hat = 0.0;
    beta2_hat = 0.5;

    # (2) Technology
    phi2 = phi1 = phi_1cap;
    A2 = A1 = A_1cap;

    if state_dependent_xi == 0
        # Constant tilting function
        scale = 1.307
        alpha_k2_hat = alpha_k1_hat = alpha_k_hat;

        # Worrisome model
        alpha_z_tilde  = alpha_z_tilde_ex;
        kappa_tilde    = kappa_hat;
        alpha_k1_tilde = alpha_k1_hat
        beta1_tilde    = beta1_hat
        alpha_k2_tilde = alpha_k2_hat
        beta2_tilde    = beta2_hat

        ell_star = ell_ex#0.026320287107624605

    elseif state_dependent_xi == 1
        # State-dependent tilting function (fixed kappa, alpha targets q)
        scale = 1.14
        alpha_k2_hat = alpha_k1_hat = alpha_k_hat + .035; #.034;

        alpha_z_tilde  = alpha_z_tilde_ex;
        kappa_tilde    = .005;
        alpha_k1_tilde = alpha_k1_hat
        beta1_tilde    = beta1_hat;
        alpha_k2_tilde = alpha_k2_hat
        beta2_tilde    = beta2_hat

        ell_star = ell_ex#0.04226404306515605

    elseif state_dependent_xi == 2
        # State-dependent tilting function (fixed beta1, alpha targets q)
        scale = 1.27
        alpha_k2_hat = alpha_k1_hat = alpha_k_hat

        alpha_z_tilde  = alpha_z_tilde_ex;
        kappa_tilde    = kappa_hat
        alpha_k1_tilde = alpha_k1_hat
        beta1_tilde    = beta1_hat + .194 #.195
        alpha_k2_tilde = alpha_k2_hat
        beta2_tilde    = beta2_hat + .194 #.195

        ell_star = ell_ex#0.06678494013273199

    end

end

sigma_k1 = [.477*sqrt(scale),               .0,   .0];
sigma_k2 = [.0              , .477*sqrt(scale),   .0];


# (3) GRID
# For analysis
if compute_irfs == 1
    II, JJ = 7001, 501;     # number of r points, number of z points
    rmax = 4.;
    rmin = -rmax;
    zmax = .7;
    zmin = -zmax;
elseif compute_irfs == 0
    II, JJ = 1001, 201;
    # II, JJ = 1401, 201;
    rmax =  18.;
    rmin = -rmax       #-25.; #-rmax;
    zmax = 1.;
    zmin = -zmax;
end

# For the optimization (over ell)
II_opt, JJ_opt = 501, 201;     # number of r points, number of z points
rmax_opt = 18.;
rmin_opt = -rmax_opt;
zmax_opt = 1.2;
zmin_opt = -zmax_opt;


# (4) Iteration parameters
maxit = 50000;        # maximum number of iterations in the HJB loop
crit  = 10e-6;      # criterion HJB loop
# Delta = 1000.;      # delta in HJB algorithm


# Initialize model objects -----------------------------------------------------
baseline = Baseline(alpha_z_hat, kappa_hat, sigma_z_1cap,
                    alpha_c_hat, beta_hat, sigma_c, delta);
baseline1 = Baseline(alpha_z_hat, kappa_hat, sigma_z,
                        alpha_k1_hat, beta1_hat, sigma_k1, delta);
baseline2 = Baseline(alpha_z_hat, kappa_hat, sigma_z,
                        alpha_k2_hat, beta2_hat, sigma_k2, delta);
technology = Technology(A_1cap, phi_1cap);
technology1 = Technology(A1, phi1);
technology2 = Technology(A2, phi2);
model = TwoCapitalEconomy(baseline1, baseline2, technology1, technology2);

worrisome = TwoCapitalWorrisome(alpha_z_tilde, kappa_tilde,
                                alpha_k1_tilde, beta1_tilde,
                                alpha_k2_tilde, beta2_tilde);
worrisome_noR = TwoCapitalWorrisome(alpha_z_hat, kappa_hat,
                                    alpha_k1_hat, beta1_hat,
                                    alpha_k2_hat, beta2_hat);

grid = Grid_rz(rmin, rmax, II, zmin, zmax, JJ);
grid_opt = Grid_rz(rmin_opt, rmax_opt, II_opt, zmin_opt, zmax_opt, JJ_opt);
params = FinDiffMethod(maxit, crit, Delta);

xi0, xi1, xi2 = tilting_function(worrisome, model);

if symmetric_returns == 0
    if state_dependent_xi == 0
        params.Delta = Delta;
    elseif state_dependent_xi == 1
        params.Delta = Delta;
    elseif state_dependent_xi == 2
        params.Delta = Delta
    end
end
#==============================================================================#
# WITH ROBUSTNESS
#==============================================================================#

println(" (3) Compute value function WITH ROBUSTNESS")
times = @elapsed begin
A, V, val, d1_F, d2_F, d1_B, d2_B, h1_F, h2_F, hz_F, h1_B, h2_B, hz_B,
        mu_1_F, mu_1_B, mu_r_F, mu_r_B, mu_z, V0, rr, zz, pii, dr, dz =
        value_function_twocapitals(ell_star, rho, fraction, model, worrisome,
                                    grid, params, symmetric_returns);
end
one_pii = 1 .- pii
println("=============================================================")
println("Convegence time (minutes): ", times/60)
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

# local uncertainty prices
h1, h2, hz = -h1_dist, -h2_dist, -hz_dist;

d1 = (policies.d1_F + policies.d1_B)/2;
d2 = (policies.d2_F + policies.d2_B)/2;
cons     = one_pii .* (model.t1.A .- d1) + pii .* (model.t2.A .- d2);

results = Dict("delta" => delta,
# Two capital stocks
"alpha_k1_hat" => alpha_k1_hat, "alpha_k2_hat" => alpha_k2_hat,
"beta1_hat" => beta1_hat, "beta2_hat" => beta2_hat,
"sigma_k1" => sigma_k1, "sigma_k2" => sigma_k2,
"sigma_z" =>  sigma_z, "A1" => A1, "A2" => A2, "phi1" => phi1, "phi2" => phi2,
"alpha_z_tilde" => alpha_z_tilde, "kappa_tilde" => kappa_tilde,
"alpha_k1_tilde" => alpha_k1_tilde, "beta1_tilde" => beta1_tilde,
"alpha_k2_tilde" => alpha_k2_tilde, "beta2_tilde" => beta2_tilde,
"xi0" => xi0, "xi1" => xi1, "xi2" => xi2,
"I" => II, "J" => JJ,
"rmax" => rmax, "rmin" => rmin, "zmax" => zmax, "zmin" => zmin,
"rr" => rr, "zz" => zz, "pii" => pii, "dr" => dr, "dz" => dz,
"maxit" => maxit, "crit" => crit, "Delta" => Delta,

"V0" => V0, "V" => V, "val" => val, "ell_star" => ell_star,

"d1" => d1, "d2" => d2,

"h1" => h1, "h2" => h2, "hz" => hz,

"cons" => cons,
"g_dist" => g_dist, "g" => g,

"times" => times,
"A_1cap" => A_1cap, "phi_1cap" => phi_1cap)
npzwrite(filename_ell*filename, results)


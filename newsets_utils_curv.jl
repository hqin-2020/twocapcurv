#==============================================================================#
#  Util functions for newsets_*.jl
#
#  Author: Balint Szoke
#  Date: Sep 2018
#==============================================================================#

using LinearAlgebra
using SparseArrays
using Interpolations
using SuiteSparse


mutable struct Baseline{T}
    a11::T
    zeta::T
    kappa::T
    sigma_z::Array{T, 1}

    beta::T
    eta::T
    sigma_k::Array{T, 1}

    delta::T
end

mutable struct Technology{T}
    alpha::T
    phi::T
end

mutable struct TwoCapitalEconomy{T}
    k1::Baseline{T}
    k2::Baseline{T}

    t1::Technology{T}
    t2::Technology{T}
end

mutable struct Worrisome{T}
    alpha_z_tilde::T
    kappa_tilde::T

    alpha_k_tilde::T
    beta_tilde::T
end

mutable struct TwoCapitalWorrisome{T}
    alpha_z_tilde::T
    kappa_tilde::T

    alpha_k1_tilde::T
    beta1_tilde::T

    alpha_k2_tilde::T
    beta2_tilde::T
end

mutable struct Grid_rz{T}
  rmin::T
  rmax::T
  I::Integer           # number of z points
  zmin::T
  zmax::T
  J::Integer           # number of z points
end

mutable struct FinDiffMethod
  maxit::Integer      # maximum number of iterations in the HJB loop
  crit::Float64       # criterion HJB loop
  Delta::Float64      # delta in HJB algorithm
end

mutable struct PolicyFunctions
  d1_F::Array{Float64,2}
  d2_F::Array{Float64,2}
  d1_B::Array{Float64,2}
  d2_B::Array{Float64,2}

  h1_F::Array{Float64,2}
  h2_F::Array{Float64,2}
  hz_F::Array{Float64,2}

  h1_B::Array{Float64,2}
  h2_B::Array{Float64,2}
  hz_B::Array{Float64,2}

end

interpol = Interpolations.Extrapolation{Float64,2,
            Interpolations.GriddedInterpolation{Float64,2,Float64,Gridded{Linear},
            Tuple{Array{Float64,1},Array{Float64,1}}}, Gridded{Linear},Throw{Nothing}}


#==============================================================================#
#  FUNCTIONS -- SINGLE CAPITAL
#==============================================================================#
function calibration(investment_capital::Float64,
                     consumption_investment::Float64,
                     consumption_mean::Float64,
                     delta::Float64,
                     sigma_c::Array{Float64,1})

    d = investment_capital;
    x = consumption_investment;
    alpha_c = consumption_mean;

    A = (1 + x)*d;
    phi = (1 - delta/(x*d))/d;
    alpha_k = alpha_c + (.01)*dot(sigma_c, sigma_c)/2 - 100*(d-(phi/2)*d^2);

    return A, phi, alpha_k
end

function calibration2(phi::Float64,
                      consumption_investment::Float64,
                      consumption_mean::Float64,
                      delta::Float64,
                      sigma_c::Array{Float64,1})

    x = consumption_investment;
    alpha_c = consumption_mean;

    #d = (1+sqrt(1 - 4*delta*phi/x))/(2*phi);
    d = (x + sqrt(x^2 - 4*delta*phi*x))/(2*phi*x);

    A = (1 + x)*d;
    alpha_k = alpha_c + (.01)*dot(sigma_c, sigma_c)/2 - 100*(d-(phi/2)*d^2);

    return A, phi, alpha_k, d
end

function calibration3(phi::Float64,
                      A::Float64,
                      delta::Float64,
                      alpha_c::Float64,
                      sigma_c::Array{Float64,1})

    d = (.5)*(A + 1/phi - sqrt((1/phi - A)^2 + 4*delta/phi))
    x = A/d -1;

    alpha_k = alpha_c + (.01)*dot(sigma_c, sigma_c)/2 - 100*(d-(phi/2)*d^2);

    return d, x, alpha_k
end


function minimum_xi(alpha_z_kappa::Float64,
                    alpha_z_hat::Float64,
                    sigma_z::Array{Float64, 1})

    if size(sigma_z)[1] == 2
        ratio = sigma_z[1]/sigma_z[end]
    else
        ratio = sqrt(2)*sigma_z[1]/sigma_z[end]
    end

    term = (3/4)/(1-ratio^2/(4*(1+ratio^2)))

    return sqrt(term)*(alpha_z_kappa - alpha_z_hat) + alpha_z_hat
end


function tilting_function(w::Worrisome{Float64},
                          b::Baseline{Float64})

    sigma_k = b.sigma_k;
    sigma_z = b.sigma_z;

    ind_sk = (sigma_k[1] == 0.) ? 2 : 1

    eta0_1 = (w.alpha_k_tilde - b.alpha_k_hat)/sigma_k[ind_sk]
    eta0_2 = (w.alpha_z_tilde - b.alpha_z_hat - sigma_z[ind_sk]*eta0_1)/sigma_z[end];

    eta1_1 = (w.beta_tilde - b.beta_hat)/sigma_k[ind_sk]
    eta1_2 = (b.kappa_hat - w.kappa_tilde - sigma_z[ind_sk]*eta1_1)/sigma_z[end];

    eta0 = [eta0_1, eta0_2];
    eta1 = [eta1_1, eta1_2];

    xi0 = dot(eta0, eta0);
    xi1 = dot(eta0, eta1);
    xi2 = dot(eta1, eta1);

    return xi0, xi1, xi2
end


function tilting_function(w::TwoCapitalWorrisome{Float64},
                          b::TwoCapitalEconomy{Float64})

    sigma_k1 = b.k1.sigma_k;
    sigma_k2 = b.k2.sigma_k;
    sigma_z = b.k1.sigma_z;


    eta0_1 = (w.alpha_k1_tilde - b.k1.alpha_k_hat)/sigma_k1[1]
    eta0_2 = (w.alpha_k2_tilde - b.k2.alpha_k_hat)/sigma_k2[2]
    eta0_3 = (w.alpha_z_tilde - b.k1.alpha_z_hat - sigma_z[1]*eta0_1 -
                                                sigma_z[2]*eta0_2)/sigma_z[3];

    eta1_1 = (w.beta1_tilde - b.k1.beta_hat)/sigma_k1[1]
    eta1_2 = (w.beta2_tilde - b.k2.beta_hat)/sigma_k2[2]
    eta1_3 = (b.k1.kappa_hat - w.kappa_tilde - sigma_z[1]*eta1_1 -
                                                sigma_z[2]*eta1_2)/sigma_z[3];

    eta0 = [eta0_1, eta0_2, eta0_3];
    eta1 = [eta1_1, eta1_2, eta1_3];

    xi0 = dot(eta0, eta0);
    xi1 = dot(eta0, eta1);
    xi2 = dot(eta1, eta1);

    return xi0, xi1, xi2
end


function dstar_singlecapital(t::Technology{Float64}, b::Baseline{Float64})
    (1 + t.A*t.phi - sqrt((1 - t.A*t.phi)^2 + 4*t.phi*b.delta))/(2*t.phi);
end



function value_function(ell::Float64,
                        z::Float64,
                        w::Worrisome{Float64},
                        b::Baseline{Float64},
                        t::Technology{Float64})

    xi0, xi1, xi2 = tilting_function(w, b);
    d_star = dstar_singlecapital(t, b);
    z_bar = b.alpha_z_hat / b.kappa_hat;

    sigma_z2 = dot(b.sigma_z, b.sigma_z);
    sigma_kz = dot(b.sigma_z, b.sigma_k);

    #-------------------------------------------#
    # Quadratic term -- W2
    #-------------------------------------------#
    square_root = sqrt((b.delta + 2*b.kappa_hat)^2 - 4*sigma_z2*xi2);
    omega2 = (b.delta + 2*b.kappa_hat - square_root)/(2*sigma_z2);
    W2 = -ell*omega2;

    if (xi0 == 0.0) && (xi1 == 0.0) && (xi2 == 0.0)
        ell_inv = .0;
        omega2, W2 = .0, .0;
    else
        ell_inv = 1. / ell;
    end

    #-------------------------------------------#
    # Linear term -- W1
    #-------------------------------------------#
    W1 = ((-ell*xi1 + (.01)*b.beta_hat + (.01)*omega2*sigma_kz)/
                                (b.delta + b.kappa_hat - omega2*sigma_z2));

    #-------------------------------------------#
    # Linear term -- W1
    #-------------------------------------------#
    kbar = 2*(d_star-t.phi*d_star^2/2+(.01)*(b.alpha_k_hat+b.beta_hat*z_bar));
    wc_d = (.01)*b.sigma_k + b.sigma_z*W1

    W0 = (2*b.delta*log(t.A-d_star) + kbar - (.01)^2*dot(b.sigma_k, b.sigma_k) -
                ell*xi0 + sigma_z2*W2 - ell_inv*dot(wc_d, wc_d))/b.delta;

    return (.5)*(W0 + 2*W1*(z-z_bar) + W2*(z-z_bar)^2), W0, W1, W2, omega2;

end


function worst_case(alpha_z_tilde::Float64,
                    kappa_tilde::Float64,
                    alpha_k_tilde::Float64,
                    beta_tilde::Float64,
                    b::Baseline{Float64},
                    t::Technology{Float64})


    zbar = b.alpha_z_hat / b.kappa_hat;
    worrisome = Worrisome(alpha_z_tilde, kappa_tilde, alpha_k_tilde, beta_tilde);

    res = optimize(r-> -value_function(r, zbar, worrisome, b, t)[1], 0.0, 10.0);
    l_star = Optim.minimizer(res);

    W0, W1, W2, w2 = value_function(l_star, zbar, worrisome, b, t)[2:end];

    Sigma = hcat((.01)*b.sigma_k, b.sigma_z);
    H_0 = -(1/l_star)*(Sigma * [1, W1]);
    H_1 = -(1/l_star)*(Sigma * [0, W2]);

    Sigma = vcat(b.sigma_k', b.sigma_z');
    dist_0 = Sigma * H_0;
    dist_1 = Sigma * H_1;

    # Calculate the parameters under the worst-case model
    alpha_z = b.alpha_z_hat + dist_0[2];
    kappa = b.kappa_hat - dist_1[2];

    alpha_k = b.alpha_k_hat + dist_0[1];
    beta = b.beta_hat + dist_1[1];

    dc = alpha_k + beta * alpha_z/kappa - b.alpha_k_hat - b.beta_hat * zbar;

    # Relative entropy
    s2 = dot(b.sigma_z, b.sigma_z);
    rho_2 = dot(H_1, H_1) / 2 / kappa;
    rho_1 = dot(H_0, H_1) / kappa + rho_2 * alpha_z / kappa;
    re = dot(H_0, H_0) / 2 + rho_1 * alpha_z + rho_2 * s2 / 2;
    q = sqrt(2*re)

    return H_0, H_1, alpha_k, beta, alpha_z, kappa, dc, re, q, l_star
end


function worst_case_statistician(alpha_z_tilde::Float64,
                                 kappa_tilde::Float64,
                                 alpha_k_tilde::Float64,
                                 beta_tilde::Float64,
                                 b::Baseline{Float64},
                                 t::Technology{Float64})




    zbar = b.alpha_z_hat / b.kappa_hat;

    worrisome = Worrisome(alpha_z_tilde, kappa_tilde, alpha_k_tilde, beta_tilde);
    xi_0, xi_1, xi_2 = tilting_function(worrisome, b);

    sigma_z2 = dot(b.sigma_z, b.sigma_z)
    sigma_kz = dot(b.sigma_z, b.sigma_k);

    # # coeffs of R2
    rho_22 = -(b.kappa_hat - sqrt(b.kappa_hat^2 - sigma_z2*xi_2))/sigma_z2

    # coeffs of R1
    denom = sqrt(b.kappa_hat^2 - sigma_z2*xi_2)
    rho_11 = -xi_1/denom
    rho_10 = (.01)*b.delta*(b.beta_hat - rho_22*sigma_kz)/denom

    # coeffs of r
    r_00 = (.01)*b.delta*(b.alpha_k_hat - rho_11*sigma_kz) - sigma_z2*rho_10*rho_11
    r_01 = (sigma_z2*rho_22 - xi_0 - sigma_z2*rho_11^2)/2
    aux = b.delta*(.01)*b.sigma_k + rho_10*b.sigma_z
    r_0m1 = -dot(aux, aux)/2

    if r_01 == 0.
        l_star = Inf
    else
        l_star = sqrt(r_0m1/r_01)
    end

    H_0 = -(1/l_star)*(b.delta * (.01)*b.sigma_k + b.sigma_z * rho_10) - b.sigma_z * rho_11
    H_1 = - b.sigma_z * rho_22

    Sigma = vcat(b.sigma_k', b.sigma_z');
    dist_0 = Sigma * H_0;
    dist_1 = Sigma * H_1;

    # Calculate the parameters under the worst-case model
    alpha_z = b.alpha_z_hat + dist_0[2];
    kappa = b.kappa_hat - dist_1[2];

    alpha_k = b.alpha_k_hat + dist_0[1];
    beta = b.beta_hat + dist_1[1];

    dc = alpha_k + beta * alpha_z/kappa - b.alpha_k_hat - b.beta_hat * zbar;

    # Relative entropy
    s2 = dot(b.sigma_z, b.sigma_z);
    rho_2 = dot(H_1, H_1) / 2 / kappa;
    rho_1 = dot(H_0, H_1) / kappa + rho_2 * alpha_z / kappa;
    re = dot(H_0, H_0) / 2 + rho_1 * alpha_z + rho_2 * s2 / 2;
    q = sqrt(2*re)

    return H_0, H_1, alpha_k, beta, alpha_z, kappa, dc, re, q, l_star
end




function choose_alpha(q::Float64,
                      kappa_tilde::Float64,
                      alpha_k_tilde::Float64,
                      beta_tilde::Float64,
                      b::Baseline{Float64},
                      t::Technology{Float64})

    a_tilde = fzero(a -> worst_case(a, kappa_tilde, alpha_k_tilde, beta_tilde, b, t)[end-1]-q, 0, -0.1);

    return a_tilde, worst_case(a_tilde, kappa_tilde, alpha_k_tilde, beta_tilde, b, t);
end


function choose_beta(q::Float64,
                     alpha_z_tilde::Float64,
                     kappa_tilde::Float64,
                     alpha_k_tilde::Float64,
                     b::Baseline{Float64},
                     t::Technology{Float64})

    beta_tilde = fzero(x -> worst_case(alpha_z_tilde, kappa_tilde, alpha_k_tilde, x, b, t)[end-1]-q,
                                    b.beta_hat, b.beta_hat + 0.22);

    return beta_tilde, worst_case(alpha_z_tilde, kappa_tilde, alpha_k_tilde, beta_tilde, b, t);
end


function choose_alpha_statistician(q::Float64,
                      kappa_tilde::Float64,
                      alpha_k_tilde::Float64,
                      beta_tilde::Float64,
                      b::Baseline{Float64},
                      t::Technology{Float64})

    a_tilde = fzero(a -> worst_case_statistician(a, kappa_tilde, alpha_k_tilde, beta_tilde, b, t)[end-1]-q, 0., -0.1);

    return a_tilde, worst_case_statistician(a_tilde, kappa_tilde, alpha_k_tilde, beta_tilde, b, t);
end


function choose_beta_statistician(q::Float64,
                     alpha_z_tilde::Float64,
                     kappa_tilde::Float64,
                     alpha_k_tilde::Float64,
                     b::Baseline{Float64},
                     t::Technology{Float64})

    beta_tilde = fzero(x -> worst_case_statistician(alpha_z_tilde, kappa_tilde, alpha_k_tilde, x, b, t)[end-1]-q,
                                    b.beta_hat, b.beta_hat + 0.22);

    return beta_tilde, worst_case_statistician(alpha_z_tilde, kappa_tilde, alpha_k_tilde, beta_tilde, b, t);
end



function chernoff_objective_single(r::Float64,
                                   H0::Array{Float64, 1},
                                   H1::Array{Float64, 1},
                                   b::Baseline{Float64})

    # Pull out useful info
    alpha_z, kappa, sigma_z = b.alpha_z_hat, b.kappa_hat, b.sigma_z
    s2 = dot(sigma_z, sigma_z);

    kappa_wc = kappa - r*dot(sigma_z, H1);
    alpha_z_wc = alpha_z + r*dot(sigma_z, H0);

    zeta_0 = - r * (r-1) * dot(H0, H0);
    zeta_1 = - r * (r-1) * dot(H0, H1);
    zeta_2 = - r * (r-1) * dot(H1, H1);

    lambda_2 = (kappa_wc-sqrt(kappa_wc^2 + zeta_2 * s2)) / s2;
    lambda_1 = - (zeta_1 - alpha_z_wc*lambda_2)/sqrt(kappa_wc^2 + zeta_2*s2);

    psi = (.5)*zeta_0 - lambda_1*alpha_z_wc - (.5)*(lambda_2 + lambda_1^2)*s2

    return - psi
end


function worrisome_entropy(alpha_z_tilde::Float64,
                           kappa_tilde::Float64,
                           alpha_k_tilde::Float64,
                           beta_tilde::Float64,
                           b::Baseline{Float64})

    eta0 = (alpha_z_tilde - b.alpha_z_hat)/b.sigma_z[end];
    eta1 = (b.kappa_hat - kappa_tilde)/b.sigma_z[end];

    s2 = dot(b.sigma_z, b.sigma_z);
    rho_2 = dot(eta1, eta1) / 2 / kappa_tilde;
    rho_1 = dot(eta0, eta1) / kappa_tilde + rho_2 * alpha_z_tilde / kappa_tilde;
    re = dot(eta0, eta0) / 2 + rho_1 * alpha_z_tilde + rho_2 * s2 / 2;

    q = sqrt(re*2);
    eta0 = [0, eta0];
    eta1 = [0, eta1];

    res = optimize(r -> chernoff_objective_single(r, eta0, eta1, b), 0., 1.);
    chernoff = - Optim.minimum(res)
    if chernoff == 0
        halflife = Inf
    else
        halflife = log(2) / chernoff
    end

    return q, re, chernoff, halflife

end


function entropy_worrisome(q::Float64,
                           kappa_tilde::Float64,
                           alpha_k_tilde::Float64,
                           beta_tilde::Float64,
                           b::Baseline{Float64})

    alpha_z_tilde = fzero(a->worrisome_entropy(a, kappa_tilde, alpha_k_tilde,
                                               beta_tilde, b)[1]-q, -1., 0.);
    return alpha_z_tilde
end




#==============================================================================#
#  FUNCTIONS -- TWO CAPITAL ECONOMIES
#==============================================================================#
function dstar_twocapitals!(d1::Array{Float64,2},
                            d2::Array{Float64,2},
                            Vr::Array{Float64,2},
                            pii::Array{Float64,2},
                            IJ::Int64,
                            fraction::Float64,
                            rho::Float64,
                            V::Array{Float64, 2},
                            model::TwoCapitalEconomy{Float64})

    # Check for the positivity of the critical term in the quad root formula
    # This might not hold if the technology parameters are weird
    alpha = model.t1.alpha; 
    zeta = model.k1.zeta;
    kappa = model.k1.kappa;
    delta = model.k1.delta;
    phi1 = model.t1.phi;
    phi2 = model.t2.phi;

    for i=1:IJ
        p, vr = pii[i], Vr[i]
        k1a = (1-zeta + zeta*exp.(p*(1-kappa))).^(1/(kappa-1));
        k2a = ((1-zeta)*exp.(p*(kappa-1)) + zeta).^(1/(kappa-1));
        
        # function f(d1x)
        #     d2_temp = (((1-zeta) * k1a.^(1-kappa)-Vr[i])*(1/(1+phi1*d1x))/(delta*exp.(V[i]*(rho-1))*k1a)).^(-1/rho);
        #     d2x = (alpha - d1x*k1a-d2_temp)/k2a;
        #     return (zeta*k2a^(1-kappa)+Vr[i])/(1+phi2*d2x)-(delta*exp.(V[i]*(rho-1))*k1a)*(alpha - d1x*k1a-d2x*k2a).^(-rho)*k2a
            
        function f(d1x)
            fc_slope = ((1-zeta)*k1a.^(1-kappa)-Vr[i])
            fi2_com = k1a*(zeta*(k2a).^(1-kappa)+Vr[i])/phi2/k2a/fc_slope
            fi2_constant = fi2_com-1/phi2
            fi2_slope = fi2_com*phi1
            d2x = fi2_slope*d1x+fi2_constant;
            return (zeta*k2a^(1-kappa)+Vr[i])/(1+phi2*d2x)-(delta*exp.(V[i]*(rho-1))*k1a)*(alpha - d1x*k1a-d2x*k2a).^(-rho)*k2a
        end
        if rho == 1.0
            c_slope = ((1-zeta)*k1a.^(1-kappa)-Vr[i])
            i2_com = k1a*(zeta*(k2a).^(1-kappa)+Vr[i])/phi2/k2a/c_slope
            i2_constant = i2_com-1/phi2
            i2_slope = i2_com*phi1
            i1_RHS_slope = -k1a*c_slope
            i2_RHS_slope = -k2a*c_slope*i2_slope
            RHS_costant = c_slope*alpha-k2a*c_slope*i2_constant
            i1_LHS_slope = phi1*delta*k1a
            i1_LHS_constant = delta*k1a
            LHS_constant = i1_LHS_constant - RHS_costant
            RHS_slope = i1_RHS_slope +i2_RHS_slope - i1_LHS_slope
            d1_root = LHS_constant/RHS_slope
            d2_root = i2_slope*d1_root+i2_constant
            # d2y = ((1-zeta) * k1a.^(1-kappa)-Vr[i])*(1/(1+phi1*d1_root))/(delta*k1a).^(-1);
            # d2_root = (alpha - d1_root*k1a-d2y)/k2a;
        else
            
            x0 = 0.03;
            d1_root = find_zero(f, x0, Roots.Order1(), maxiters = 100000000, xatol = 10e-12, xrtol = 10e-12, atol = 0, rtol = 0,  strict = true);
            d2y = (((1-zeta) * k1a.^(1-kappa)-Vr[i])*(1/(1+phi1*d1_root))/(delta*exp.(V[i]*(rho-1))*k1a)).^(-1/rho);
            d2_root = (alpha - d1_root*k1a-d2y)/k2a;
            
        end

        # function f(d1x)
        #     d2_temp = (((1-zeta) * k1a.^(1-kappa)-Vr[i])*(1/(1+phi1*d1x))/(delta*exp.(V[i]*(rho-1))*k1a)).^(-1/rho);
        #     d2x = (alpha - d1x*k1a-d2_temp)/k2a;
        #     return (zeta*k2a.^(1-kappa)+Vr[i])/(1+phi2*d2x)-(delta*exp.(V[i]*(rho-1))*k2a)*(alpha - d1x*k1a-d2x*k2a).^(-rho)
        # end
        # x0 = 0.02;
        # d1_root_c = find_zero(f, x0, Roots.Order1(), maxiters = 100000000, strict = true);
        # d2y = (((1-zeta) * k1a.^(1-kappa)-Vr[i])*(1/(1+phi1*d1_root_c))/(delta*exp.(V[i]*(rho-1))*k1a)).^(-1/rho);
        # d2_root_c = (alpha - d1_root_c*k1a-d2y)/k2a;

        # function f(d1x)
        #     d2_temp = (((1-zeta) * k1a.^(1-kappa)-Vr[i])*(1/(1+phi1*d1x))/(delta*k1a)).^(-1);
        #     d2x = (alpha - d1x*k1a-d2_temp)/k2a;
        #     return (zeta*k2a.^(1-kappa)+Vr[i])/(1+phi2*d2x)-(delta*k2a)*(alpha - d1x*k1a-d2x*k2a).^(-1)
        # end
        # x0 = 0.03;
        # d1_root_c = find_zero(f, x0, Roots.Order1(), maxiters = 100000000, strict = true);
        # d1_root_d = find_zero(f, x0, Roots.Order1(), maxiters = 10000000000, xatol = 10e-18, xrtol = 10e-18, atol = 10e-18, rtol = 10e-18, strict = true);
        # d2y = (((1-zeta) * k1a.^(1-kappa)-Vr[i])*(1/(1+phi1*d1_root_c))/(delta*k1a)).^(-1);
        # d2_root_c = (alpha - d1_root_c*k1a-d2y)/k2a;
        if i <10
            println("root: ", d1_root, ", analytical root: ", f(d1_root))
        #     println("1:", d1_root, ", 2:", d1_root_c,  ", 3:",d1_root_d, ", analytical root: ", f(d1_root),  ", numerical root: ",f(d1_root_c),  ", numerical root strict: ",f(d1_root_d))
        end
        # println(d1_root - d1_root_c)
        # println(d2_root-d2_root_c)
        d1[i] = d1_root;
        d2[i] = d2_root;
        # d1old = copy(d1[i]);
        # d2old = copy(d2[i]);
        # if d1old<0
        #     d1old = 0.001;
        # end
        # if d2old<0
        #     d2old = 0.001;
        # end

        # c = alpha - d1old*k1a - d2old*k2a;
        # if c<0
        #     c= 0.001
        # end
        # d1new_temp = ((1-zeta)*k1a^(1-kappa)-Vr[i]) / (delta*(exp.(rho-1)*V[i])*c^(-rho)*k1a);
        # d1new = (d1new_temp-1)/phi1;
        # d2new_temp = (zeta*k2a^(1-kappa)+Vr[i]) / (delta*(exp.(rho-1)*V[i])*c^(-rho)*k2a);
        # d2new = (d2new_temp-1)/phi2;
        # if d1new<0
        #     d1new = 0.001;
        # end
        # if d2new<0
        #     d2new = 0.001;
        # end
        

        # if d1new*k1a>alpha*0.36
        #     d1new = alpha*0.36/k1a-0.001;
        # elseif d1new<0
        #     d1new = 0.001;
        # end

        # if d2new*k2a>alpha*0.36
        #     d2new = alpha*0.36/k2a-0.001;
        # elseif d1new<0
        #     d2new = 0.001;
        # end

        # d1_temp = d1new * fraction + d1old *(1-fraction);
        # d2_temp = d2new * fraction + d2old *(1-fraction);
        # d1[i] = d1_temp;
        # d2[i] = d2_temp;
        # if d1_temp<0
        #     d1[i] = 0.001;
        # else
        #     d1[i] = d1_temp;
        # end
        # if d2_temp<0
        #     d2[i] = 0.001;
        # else
        #     d2[i] = d2_temp;
        # end
        
        # if d1_temp*k1a>alpha*0.36
        #     d1[i] = alpha*0.36/k1a-0.001;
        # elseif d1_temp<0
        #     d1[i] = 0.001;
        # else
        #     d1[i] = d1_temp;
        # end

        # if d2_temp*k2a>alpha*0.36
        #     d2[i] = alpha*0.36/k2a-0.001;
        # elseif d2_temp<0
        #     d2[i] = 0.001;
        # else
        #     d2[i] = d2_temp;
        # end
    end

    nothing
end


function upwind_transform!(var::Array{Float64, 2},
                           var_F::Array{Float64, 2},
                           var_B::Array{Float64, 2},
                           drift_F::Array{Float64, 2},
                           drift_B::Array{Float64, 2},
                           IJ::Int64,
                           dim::Integer=1)

    If = drift_F .>= 0.;
    Ib = .!If .& (drift_B .< 0.);

    if dim ==1
      If[1, :] .= 1.;    # force to use forward at the first row
      If[end, :] .= 0.;
      Ib[end, :] .= 1.;  # force to use backward at the last row
      Ib[1, :] .= 0.;
    elseif dim==2
      If[:, 1] .= 1.;    # force to use forward at the first col
      If[:, end] .= 0.;
      Ib[:, end] .= 1.;  # force to use backward at the last col
      Ib[:, 1] .= 0.;
    end

    for i=1:IJ
        I0 = (If[i] + Ib[i]) == 0;  # when (drift_F < 0 and drift_B > 0)
        var[i] = If[i]*var_F[i] + Ib[i]*var_B[i] + I0*var_F[i];
    end

    nothing
end



function drifts!(mu_1::Array{Float64, 2},
                 mu_r::Array{Float64, 2},
                 d1::Array{Float64, 2},
                 d2::Array{Float64, 2},
                 zz::Array{Float64, 2},
                 pii::Array{Float64, 2},
                 IJ::Int64,
                 model::TwoCapitalEconomy)

    phi1, phi2 = model.t1.phi, model.t2.phi;

    zeta = model.k1.zeta;
    kappa = model.k1.kappa;
    beta1 = model.k1.beta;
    beta2 = model.k2.beta;
    eta1 = model.k1.eta;
    eta2 = model.k2.eta;
    s_k1 = model.k1.sigma_k;
    s_k2 = model.k2.sigma_k;

    for i=1:IJ
        p, z = pii[i], zz[i];
        k1a = (1-zeta + zeta*exp.(p*(1-kappa))).^(1/(kappa-1));
        k2a = ((1-zeta)*exp.(p*(kappa-1)) + zeta).^(1/(kappa-1));
        
        dkadk1dk1 = (kappa-1) * ((1-zeta).^2*(k1a).^(-2*kappa+2) - kappa/(kappa-1)*(1-zeta)*(k1a).^(-kappa+1));
        dkadk1dk2 = (kappa-1) * zeta*(1-zeta)*(k1a).^(-kappa+1)*(k2a).^(-kappa+1);
        dkadk2dk2 = (kappa-1) * (zeta.^2*(k2a).^(-2*kappa+2) - kappa/(kappa-1)*(1-zeta)*(k2a).^(-kappa+1));

        mu_k1 = 1/phi1 * log(1+phi1*d1[i]) + beta1*z - eta1;
        mu_k2 = 1/phi2 * log(1+phi2*d2[i]) + beta2*z - eta2;

        mu_r[i] = mu_k2 - mu_k1 - (1/2)*(dot(s_k2,s_k2) - dot(s_k1,s_k1)) 

        mu_1[i] = mu_k1*(1-zeta)*(k1a).^(1-kappa)+
                    mu_k2*(zeta)*(k2a).^(1-kappa)+
                    1/2*(dot(s_k1,s_k1)*dkadk1dk1 + dot(s_k2,s_k2)*dkadk2dk2 + 2*dot(s_k1,s_k2)*dkadk1dk2)  
    end

    nothing

end


function drifts_distortion!(h::Array{Float64, 2},
                            s_k1::Float64,
                            s_k2::Float64,
                            s_z::Float64,
                            IJ::Int64,
                            pii::Array{Float64, 2},
                            Vr::Array{Float64, 2},
                            Vz::Array{Float64, 2},
                            model::TwoCapitalEconomy)

    zeta = model.k1.zeta;
    kappa = model.k1.kappa;

    for i=1:IJ
        p = pii[i];
        k1a = (1-zeta + zeta*exp.(p*(1-kappa))).^(1/(kappa-1));
        k2a = ((1-zeta)*exp.(p*(kappa-1)) + zeta).^(1/(kappa-1));

        h[i] = (s_k1*(1-zeta)*(k1a).^(1-kappa) + s_k2*(zeta)*(k2a).^(1-kappa) + (s_k2-s_k1)*Vr[i]) + s_z*Vz[i];
    end

    nothing
end


function create_uu!(uu::Array{Float64, 1},
                    gamma::Float64,
                    rho::Float64,
                    d1::Array{Float64, 2},
                    d2::Array{Float64, 2},
                    h1::Array{Float64, 2},
                    h2::Array{Float64, 2},
                    hz::Array{Float64, 2},
                    mu_1::Array{Float64, 2},
                    pii::Array{Float64, 2},
                    zz::Array{Float64, 2},
                    IJ::Int64,
                    V::Array{Float64, 2},
                    model::TwoCapitalEconomy)

    alpha = model.t1.alpha;
    delta = model.k1.delta;
    zeta = model.k1.zeta;
    kappa = model.k1.kappa;

    for i=1:IJ
        p, z = pii[i], zz[i]
        k1a = (1-zeta + zeta*exp.(p*(1-kappa))).^(1/(kappa-1));
        k2a = ((1-zeta)*exp.(p*(kappa-1)) + zeta).^(1/(kappa-1));

        c = alpha - d1[i]*k1a - d2[i]*k2a;
        # if c<0
        #     c= 0.001
        # end
        penalty_term = (1-gamma)*(h1[i]^2 + h2[i]^2 + hz[i]^2)/2;
        if rho == 1.0
            uu[i] = (delta*log(c) + penalty_term + mu_1[i]);
        else
            uu[i] = (delta/(1-rho)*(c.^(1-rho)*exp.((rho-1)*V[i])-1)+ penalty_term + mu_1[i]);
        end
    end

    nothing
end



function create_Aval!(Aval, d_, c_, e_, b_, f_, a_1, a_2, II, JJ)
    # A = spdiagm( 0 => d_[1:IJ],
    #              1 => c_[1:IJ-1],
    #             -1 => e_[2:IJ],
    #             II => b_[1:II*(JJ-1)],
    #            -II => f_[II+1:end],
    #          -II-1 => a_1[II+2:end],
    #           II+1 => a_2[1:II*(JJ-1)-1]);

    # A = spdiagm( 0 => ones(II*JJ),
    #                1 => ones(II*JJ-1),
    #               -1 => ones(II*JJ-1),
    #               II => ones(II*(JJ-1)),
    #              -II => ones(II*(JJ-1)),
    #            -II-1 => ones(II*(JJ-1)-1),
    #             II+1 => ones(II*(JJ-1)-1));


    iter = 1
    Aval[iter] = d_[1]; iter += 1;
    Aval[iter] = e_[1+1]; iter += 1;
    Aval[iter] = f_[II+1]; iter += 1;
    Aval[iter] = a_1[II+1+1]; iter += 1;

    for j=2:II
        Aval[iter] = c_[-1+j]; iter += 1;
        Aval[iter] = d_[j]; iter += 1;
        Aval[iter] = e_[1+j]; iter += 1;
        Aval[iter] = f_[II+j]; iter += 1;
        Aval[iter] = a_1[II+1+j]; iter += 1;
    end

    Aval[iter] = b_[-II+II+1]; iter += 1;
    Aval[iter] = c_[-1+II+1]; iter += 1;
    Aval[iter] = d_[II+1]; iter += 1;
    Aval[iter] = e_[1+II+1]; iter += 1;
    Aval[iter] = f_[II+II+1]; iter += 1;
    Aval[iter] = a_1[II+1+II+1]; iter += 1;

    for j=(II+2):(II*(JJ-1)-1)
        Aval[iter] = a_2[-(II+1)+j]; iter += 1;
        Aval[iter] = b_[-II+j]; iter += 1;
        Aval[iter] = c_[-1+j]; iter += 1;
        Aval[iter] = d_[j]; iter += 1;
        Aval[iter] = e_[1+j]; iter += 1;
        Aval[iter] = f_[II+j]; iter += 1;
        Aval[iter] = a_1[II+1+j]; iter += 1;
    end

    Aval[iter] = a_2[-(II+1)+II*(JJ-1)]; iter += 1;
    Aval[iter] = b_[-II+II*(JJ-1)]; iter += 1;
    Aval[iter] = c_[-1+II*(JJ-1)]; iter += 1;
    Aval[iter] = d_[II*(JJ-1)]; iter += 1;
    Aval[iter] = e_[1+II*(JJ-1)]; iter += 1;
    Aval[iter] = f_[II+II*(JJ-1)]; iter += 1;

    for j=(II*(JJ-1)+1):(II*JJ-1)
        Aval[iter] = a_2[-(II+1)+j]; iter += 1;
        Aval[iter] = b_[-II+j]; iter += 1;
        Aval[iter] = c_[-1+j]; iter += 1;
        Aval[iter] = d_[j]; iter += 1;
        Aval[iter] = e_[1+j]; iter += 1;
    end

    Aval[iter] = a_2[-(II+1)+(II*JJ)]; iter += 1;
    Aval[iter] = b_[-II+(II*JJ)]; iter += 1;
    Aval[iter] = c_[-1+(II*JJ)]; iter += 1;
    Aval[iter] = d_[(II*JJ)]; iter += 1;

    nothing
end



function value_function_twocapitals(gamma::Float64,
                                    rho::Float64,
                                    fraction::Float64,
                                    model::TwoCapitalEconomy,
                                    grid::Grid_rz,
                                    params::FinDiffMethod,
                                    symmetric::Integer=1)

      rmin, rmax, II = grid.rmin, grid.rmax, grid.I;
      zmin, zmax, JJ = grid.zmin, grid.zmax, grid.J;

      # Derived indexes
      IJ = II*JJ;
      II_half = convert(Integer, round(II/2));
      JJ_half = convert(Integer, round(JJ/2));

      maxit  = params.maxit;       # max number of iterations in the HJB loop
      crit = params.crit;          # criterion HJB loop
      Delta = params.Delta;        # delta in HJB algorithm

      r = range(rmin, stop=rmax, length=II);    # capital ratio vector
      dr = (rmax - rmin)/(II-1);
      z = range(zmin, stop=zmax, length=JJ)';   # productivity vector
      dz = (zmax - zmin)/(JJ-1);
      dr2, dz2, drdz = dr*dr, dz*dz, dr*dz;

      println("rmax = ", rmax, ", rmin = ",rmin, ", rlength = ", II)
      println("zmax = ", zmax, ", zmin = ",zmin, ", zlength = ", JJ)

      rr = r * ones(1, JJ);
      zz = ones(II, 1) * z;
      pii = rr;
    #   k1a = (1-zeta + zeta*exp.(p*(1-kappa))).^(1/(kappa-1));
    #   k2a = ((1-zeta)*exp.(p((kappa-1))) + zeta).^(1/(kappa-1));

      #========================================================================#
      # Storing matrices
      #========================================================================#
      # Value function and forward/backward finite difference matrices
      V, V0 = zeros(II, JJ), zeros(II, JJ);
      c = zeros(II, JJ);


      # These matrices need to compute choices
      Vr_F, Vr_B = zeros(II, JJ), zeros(II, JJ);
      Vz_F, Vz_B = zeros(II, JJ), zeros(II, JJ);
      Vr, Vz = zeros(II, JJ), zeros(II, JJ);

      # Choice variables (capital ratio and worst-case drift)
      d1_F, d2_F = zeros(II, JJ), zeros(II, JJ);
      d1_B, d2_B = zeros(II, JJ), zeros(II, JJ);
      h1_F, h1_B  = zeros(II, JJ), zeros(II, JJ);
      h2_F, h2_B  = zeros(II, JJ), zeros(II, JJ);
      hz_F, hz_B  = zeros(II, JJ), zeros(II, JJ);

      # Choice variables (capital ratio and worst-case drift)
      d1, d2 = zeros(II, JJ), zeros(II, JJ);
      h1, h2, hz = zeros(II, JJ), zeros(II, JJ), zeros(II, JJ);

      # Drifts
      mu_1_F, mu_1_B, mu_1 = zeros(II, JJ), zeros(II, JJ), zeros(II, JJ);
      mu_r_F, mu_r_B = zeros(II, JJ), zeros(II, JJ);
      mu_z = zeros(II, JJ);

      uu = zeros(II*JJ);
      A = spdiagm( 0 => ones(II*JJ),
                   1 => ones(II*JJ-1),
                  -1 => ones(II*JJ-1),
                  II => ones(II*(JJ-1)),
                 -II => ones(II*(JJ-1)),
               -II-1 => ones(II*(JJ-1)-1),
                II+1 => ones(II*(JJ-1)-1));
      Aval = zeros(nnz(A))

      #========================================================================#
      # MODEL PARAMETERS                                                       #
      #========================================================================#
      delta = model.k1.delta;

      # (1) Baseline model
      a11 = model.k1.a11;
      zeta = model.k1.zeta;
      kappa = model.k1.kappa;

      eta1 = model.k1.eta;
      eta2 = model.k2.eta;
      beta1 = model.k1.beta;
      beta2 = model.k2.beta;

      s_k1 = model.k1.sigma_k;
      s_k2 = model.k2.sigma_k;
      s_z =  model.k1.sigma_z;

      # (2) Two capitals case: technology
      alpha = model.t1.alpha;
      phi1, phi2 = model.t1.phi, model.t2.phi;

      t1 = dot(s_k2 - s_k1, s_k2 - s_k1)/(2*dr2);
      t2 = dot(s_k2 - s_k1, s_z)/(2*drdz);
      t3 = dot(s_z, s_z)/(2*dz2);

      #========================================================================#
      # INITIALIZATION                                                         #
      #========================================================================#
      for j=1:JJ
         V0[:, j] = range(-1, stop=-1, length=II);
      end

      v = copy(V0);
      distance = zeros(maxit);

      #========================================================================#
      # HAMILTON-JACOBI-BELLMAN EQUATION
      #========================================================================#
      mu_z = -a11*zz;
      if rho == 1.0
        I_delta = sparse((1/Delta + delta)*I, IJ, IJ);
      else
        I_delta = sparse((1/Delta)*I, IJ, IJ);
      end


      for n=1:maxit

          V = copy(v);
          V_stacked = vec(V);

          # forward diff (last row never used - known value function there)
          # backward diff (first row never used - known value function there)
          # Diff in the z dimension: 1st/last col = 0 imposed
          Vr_B[2:II, :] = Vr_F[1:II-1, :] = (V[2:II, :] - V[1:II-1, :])./dr;
          Vz_B[:, 2:JJ] = Vz_F[:, 1:JJ-1] = (V[:, 2:JJ] - V[:, 1:JJ-1])./dz;

          # Investment-capital ratios
          dstar_twocapitals!(d1_F, d2_F, Vr_F, pii, IJ, fraction, rho, V, model);
          dstar_twocapitals!(d1_B, d2_B, Vr_B, pii, IJ, fraction, rho, V, model);

          # Drifts
          drifts!(mu_1_F, mu_r_F, d1_F, d2_F, zz, pii, IJ, model);
          drifts!(mu_1_B, mu_r_B, d1_B, d2_B, zz, pii, IJ, model);

          #--------------------------------------------------------------------
          # Worst-case drifts and utility using upwind adjusted variables
          #--------------------------------------------------------------------
          upwind_transform!(Vr, Vr_F, Vr_B, mu_r_F, mu_r_B, IJ);
          upwind_transform!(Vz, Vz_F, Vz_B, mu_z  , mu_z  , IJ, 2);
          upwind_transform!(d1, d1_F, d1_B, mu_r_F, mu_r_B, IJ);
          upwind_transform!(d2, d2_F, d2_B, mu_r_F, mu_r_B, IJ);
	      upwind_transform!(mu_1, mu_1_F, mu_1_B, mu_r_F, mu_r_B, IJ);

          # WORST-CASE DISTORTIONS
          drifts_distortion!(h1, s_k1[1], s_k2[1], s_z[1], IJ, pii, Vr, Vz, model);
          drifts_distortion!(h2, s_k1[2], s_k2[2], s_z[2], IJ, pii, Vr, Vz, model);
          drifts_distortion!(hz, s_k1[3], s_k2[3], s_z[3], IJ, pii, Vr, Vz, model);

          drifts_distortion!(h1_F, s_k1[1],s_k2[1],s_z[1],IJ,pii,Vr_F,Vz_F, model);
          drifts_distortion!(h2_F, s_k1[2],s_k2[2],s_z[2],IJ,pii,Vr_F,Vz_F, model);
          drifts_distortion!(hz_F, s_k1[3],s_k2[3],s_z[3],IJ,pii,Vr_F,Vz_F, model);
          drifts_distortion!(h1_B, s_k1[1],s_k2[1],s_z[1],IJ,pii,Vr_B,Vz_B, model);
          drifts_distortion!(h2_B, s_k1[2],s_k2[2],s_z[2],IJ,pii,Vr_B,Vz_B, model);
          drifts_distortion!(hz_B, s_k1[3],s_k2[3],s_z[3],IJ,pii,Vr_B,Vz_B, model);


	      # FLOW TERM
          if symmetric==1
            create_uu!(uu, gamma, rho, d1_F, d2_F, h1_F, h2_F, hz_F, mu_1_F, pii, zz, IJ, V, model);
          elseif symmetric==0
            create_uu!(uu, gamma, rho, d1, d2, h1, h2, hz, mu_1, pii, zz, IJ, V, model);
          end
          

          #CONSTRUCT MATRIX A
          a_1 = ones(II, JJ)*t2;
          a_2 = ones(II, JJ)*t2;
          b_ = max.(mu_z, 0.)/dz .+ t3 .- t2;
          c_ = max.(mu_r_F, 0.)/dr .+ t1 .- t2;
          d_ = (-max.(mu_r_F, 0.)/dr + min.(mu_r_B, 0.)/dr - max.(mu_z, 0.)/dz +
                 min.(mu_z, 0.)/dz .- 2*(t1 + t3 - t2));
          e_ = -min.(mu_r_B, 0.)/dr .+ t1 .- t2;
          f_ = -min.(mu_z, 0.)/dz .+ t3 .- t2;

          # Adding reflection boundary in I dimension
          f_[1 , :] += a_1[1, :];
          a_1[1, :] .= 0.0;
          b_[end, :] += a_2[end, :];
          a_2[end, :] .= 0.0;

          d_[1, :] += e_[1, :];
          e_[1, :] .= 0.0;
          d_[end, :] += c_[end, :];
          c_[end, :] .= 0.0;

          # Adding reflection boundary in J dimension
          d_[:, 1] += f_[:, 1];
          e_[:, 1] += a_1[:, 1];
          d_[:, end] += b_[:, end];
          c_[:, end] += a_2[:, end];

          create_Aval!(Aval, d_, c_, e_, b_, f_, a_1, a_2, II, JJ)
          A.nzval .= Aval;

          #Cblas function for y:=a*x + y (in our case: uu^n + v^n -> uu^n)
          BLAS.axpy!(IJ, 1/Delta, V_stacked, 1, uu, 1);

          # TIME CONSUMING PART
          V_stacked = (I_delta - A) \ uu
          #ldiv!(V_stacked, factorize(I_delta - A), uu);
          #lsmr!(V_stacked, I_delta - A, uu, atol=1e-8, btol=1e-8)

          Vchange = reshape(V_stacked, II, JJ) - v;
          distance[n] = maximum(abs.(Vchange));

          v = reshape(V_stacked, II, JJ);

          println("----------------------------------");
          println("Iteration = ",n);
          println("Distance = ",distance[n]);
          println("v max = ",maximum(v));
          println("v min = ",minimum(v));
          println("d1 max = ",maximum(d1));
          println("d1 min = ",minimum(d1));
          println("d1F max = ",maximum(d1_F));
          println("d1F min = ",minimum(d1_F));
          println("d1B max = ",maximum(d1_B));
          println("d1B min = ",minimum(d1_B));
          println("d2 max = ",maximum(d2));
          println("d2 min = ",minimum(d2));
          println("d2F max = ",maximum(d2_F));
          println("d2F min = ",minimum(d2_F));
          println("d2B max = ",maximum(d2_B));
          println("d2B min = ",minimum(d2_B));
          println("h1 max = ",maximum(h1));
          println("h1 min = ",minimum(h1));
          println("h2 max = ",maximum(h2));
          println("h2 min = ",minimum(h2));
          println("hz max = ",maximum(hz));
          println("hz min = ",minimum(hz));
          println("----------------------------------")
          
          if distance[n]<crit
              val = v[II_half, JJ_half];
              #println(val)

              println("    Value Function Converged, Iteration = ", n,
                      " with gamma=", gamma,
                      " with rho=", rho,
                      " and v=", val);
              create_Aval!(Aval, d_, c_, e_, b_, f_, a_1, a_2, II, JJ)
              A.nzval .= Aval;
              break
          end
      end

    val = v[II_half, JJ_half];     # Value at (r_0, z_0)  (objective)

    return A, v,val, d1_F, d2_F, d1_B, d2_B, h1_F, h2_F, hz_F, h1_B, h2_B, hz_B,
           mu_1_F, mu_1_B, mu_r_F, mu_r_B, mu_z, c, V0, rr, zz, pii, dr, dz;
end



function stationary_distribution(A::SparseMatrixCSC{Float64,Int64},
                                 grid::Grid_rz)

      #=======================================================#
      # Construct grid
      #=======================================================#
      rmin, rmax, I = grid.rmin, grid.rmax, grid.I;
      zmin, zmax, J = grid.zmin, grid.zmax, grid.J;
      r = range(rmin, stop=rmax, length=I);    # capital ratio vector
      dr = (rmax - rmin)/(I-1);
      z = range(zmin, stop=zmax, length=J)';   # productivity vector
      dz = (zmax - zmin)/(J-1);
      rr = r * ones(1, J);
      pii = exp.(rr) ./ (1 .+ exp.(rr));


      b = zeros(I*J);
      AT = copy(A');

      #need to fix one value, otherwise matrix is singular
      i_fix = 1;
      b[i_fix] = .1;
      for j = 1:I*J
        AT[i_fix, j] = 0;
      end
      AT[i_fix, i_fix] = 1;

      #Solve linear system
      gg = AT \ b[:, 1];
      g_sum = gg' * ones(I * J, 1) * dr * dz;
      gg = gg./g_sum;

      g = reshape(gg, I, J);

      # Change of VARIABLES
      # k := f(r) = exp(r)/(1 + exp(r)) => |df^{-1}(k)/dk|
      # dlog(k/(1-k))/dk = 1/(k(1-k))
      adj = 1 ./ (pii .* (1 .- pii));

      g_dist = g .* adj;

      return g_dist, g
end


function worstcase_distortion(h1::Array{Float64,2},
                              h2::Array{Float64,2},
                              hz::Array{Float64,2},
                              pii::Array{Float64,2},
                              model::TwoCapitalEconomy)

    sigma_k1 = model.k1.sigma_k;
    sigma_k2 = model.k2.sigma_k;
    sigma_z =  model.k1.sigma_z;

    WCDist_1 = (.01)*((sigma_k1[1]*(1 .-pii) + pii*sigma_k2[1]) .* h1 +
                      (sigma_k1[2]*(1 .-pii) + pii*sigma_k2[2]) .* h2 +
                      (sigma_k1[3]*(1 .-pii) + pii*sigma_k2[3]) .* hz);

    WCDist_r = (.01)*((sigma_k2[1] - sigma_k1[1]) .* h1 +
                      (sigma_k2[2] - sigma_k1[2]) .* h2 +
                      (sigma_k2[3] - sigma_k1[3]) .* hz);

    WCDist_z = sigma_z[1] * h1 + sigma_z[2] * h2 + sigma_z[3] * hz;

    return WCDist_1, WCDist_r, WCDist_z
end



function Kolmogorov_FinDiff(policies::PolicyFunctions,
                            model::TwoCapitalEconomy,
                            grid::Grid_rz,
                            params::FinDiffMethod,
                            gamma::Float64=1.0)

      rmin, rmax, I = grid.rmin, grid.rmax, grid.I;
      zmin, zmax, J = grid.zmin, grid.zmax, grid.J;
      maxit  = params.maxit;       # max number of iterations in the HJB loop
      crit = params.crit;          # criterion HJB loop
      Delta = params.Delta;        # delta in HJB algorithm

      #VARIABLES
      r = range(rmin, stop=rmax, length=I);    # capital ratio vector
      dr = (rmax - rmin)/(I-1);
      z = range(zmin, stop=zmax, length=J)';   # productivity vector
      dz = (zmax - zmin)/(J-1);
      dr2 = dr * dr;
      dz2 = dz * dz;
      drdz = dr * dz;

      rr = r * ones(1, J);
      zz = ones(I, 1) * z;
      pii = exp.(rr) ./ (1 .+ exp.(rr));

      #========================================================================#
      # Storing matrices                                                       #
      #========================================================================#

      # Drifts
      mu_1_F = zeros(I, J);
      mu_r_F = zeros(I, J);
      mu_z = zeros(I, J);
      mu_1_B = zeros(I, J);
      mu_r_B = zeros(I, J);

      A = spzeros(I*J, I*J);

      # Various storing matrices to simplify algebra
      mu_k1_F = zeros(I, J);
      mu_k2_F = zeros(I, J);
      mu_k1_B = zeros(I, J);
      mu_k2_B = zeros(I, J);

      a_1 = zeros(I, J);
      a_2 = zeros(I, J);
      b_ = zeros(I, J);
      c_ = zeros(I, J);
      d_ = zeros(I, J);
      e_ = zeros(I, J);
      f_ = zeros(I, J);

      #========================================================================#
      # Load model parameters                                                  #
      #========================================================================#
      delta = model.k1.delta;

      # (1) Baseline model
      alpha_z_hat = model.k1.alpha_z_hat;
      kappa_hat = model.k1.kappa_hat;
      zbar = alpha_z_hat/kappa_hat;

      alpha_k1_hat = model.k1.alpha_k_hat;
      alpha_k2_hat = model.k2.alpha_k_hat;
      beta1_hat = model.k1.beta_hat;
      beta2_hat = model.k2.beta_hat;

      sigma_k1 = model.k1.sigma_k;
      sigma_k2 = model.k2.sigma_k;
      sigma_z =  model.k1.sigma_z;

      # (2) Two capitals case: technology
      A1, A2 = model.t1.A, model.t2.A;
      phi1, phi2 = model.t1.phi, model.t2.phi;

      t1 = (.01)^2 * dot(sigma_k2 - sigma_k1, sigma_k2 - sigma_k1) / (2 * dr2);
      t2 = (.01) * dot(sigma_k2 - sigma_k1, sigma_z) / (2 * drdz);
      t3 = dot(sigma_z, sigma_z) / (2 * dz2);

      #------------------------------------------------------------------------#
      # KOLMOGOROV FORWARD
      #------------------------------------------------------------------------#
      d1_F, d2_F, d1_B, d2_B = policies.d1_F, policies.d2_F, policies.d1_B, policies.d2_B;
      h1_F, h2_F, hz_F = policies.h1_F, policies.h2_F, policies.hz_F;
      h1_B, h2_B, hz_B = policies.h1_B, policies.h2_B, policies.hz_B;

      # Drifts for Forward diff
      WCDist_1_F, WCDist_r_F, WCDist_z_F = worstcase_distortion(h1_F, h2_F, hz_F, pii, model);
      mu_k1_F = d1_F .- phi1 * d1_F .^2 / 2 .+ (.01) * (alpha_k1_hat .+ beta1_hat * zz);
      mu_k2_F = d2_F .- phi2 * d2_F .^2 / 2 .+ (.01) * (alpha_k2_hat .+ beta2_hat * zz);
      mu_r_F = mu_k2_F .- mu_k1_F .- ((.01)^2/2)*(dot(sigma_k2, sigma_k2) .- dot(sigma_k1, sigma_k1)) .+ gamma * WCDist_r_F;
      mu_z_F = - kappa_hat * zz .+ gamma * WCDist_z_F;

      # Drifts for Backward diff
      WCDist_1_B, WCDist_r_B, WCDist_z_B = worstcase_distortion(h1_B, h2_B, hz_B, pii, model);
      mu_k1_B = d1_B .- phi1 * d1_B .^2 / 2 .+ (.01) * (alpha_k1_hat .+ beta1_hat * zz);
      mu_k2_B = d2_B .- phi2 * d2_B .^2 / 2 .+ (.01) * (alpha_k2_hat .+ beta2_hat * zz);
      mu_r_B = mu_k2_B .- mu_k1_B .- ((.01)^2/2)*(dot(sigma_k2, sigma_k2) .- dot(sigma_k1, sigma_k1)) .+ gamma * WCDist_r_B;
      mu_z_B = - kappa_hat * zz .+ gamma * WCDist_z_B;

      #CONSTRUCT MATRIX A
      a_1 = ones(I, J) * t2;
      a_2 = ones(I, J) * t2;
      b_ = max.(mu_z_F, 0.) / dz .+ t3 .- t2;
      c_ = max.(mu_r_F, 0.) / dr .+ t1 .- t2;
      d_ = - max.(mu_r_F, 0.) / dr .+ min.(mu_r_B, 0.) / dr .- max.(mu_z_F, 0.) / dz .+ min.(mu_z_B, 0.) / dz .- 2*(t1 .+ t3 .- t2);
      e_ = - min.(mu_r_B, 0.) / dr .+ t1 .- t2;
      f_ = - min.(mu_z_B, 0.) / dz .+ t3 .- t2;

      # Adding reflection boundary in I dimension
      f_[1 , :] += a_1[1, :];
      a_1[1, :] .= 0.0;
      b_[end, :] += a_2[end, :];
      a_2[end, :] .= 0.0;

      d_[1, :] += e_[1, :];
      e_[1, :] .= 0.0;
      d_[end, :] += c_[end, :];
      c_[end, :] .= 0.0;

      # Adding reflection boundary in J dimension
      d_[:, 1] += f_[:, 1];
      e_[:, 1] += a_1[:, 1];
      d_[:, end] += b_[:, end];
      c_[:, end] += a_2[:, end];

      A = spdiagm( 0 => reshape(d_ , I*J)) +
          spdiagm( 1 => reshape(c_ , I*J)[1:I*J-1]) +
          spdiagm(-1 => reshape(e_ , I*J)[2:I*J]) +
          spdiagm( I => reshape(b_ , I*J)[1:I*(J-1)]) +
          spdiagm(-I => reshape(f_ , I*J)[I+1:end]) +
          spdiagm(-I-1 => reshape(a_1, I*J)[I+2:end]) +
          spdiagm(I+1 => reshape(a_2, I*J)[1:I*(J-1)-1]);

      #------------------------------------------------------------------------
      # When gamma is provided -> compute findiff matrix for Chernoff entropy
      #------------------------------------------------------------------------
      if (gamma != 1.0)
          # centered finite difference for worst-case
          h1 = (h1_F + h1_B)/2;
          h2 = (h2_F + h2_B)/2;
          hz = (hz_F + hz_B)/2;

          flow = (- gamma + gamma^2) * (h1 .^ 2 + h2 .^ 2 + hz .^ 2) / 2;
          A += spdiagm(0 => reshape(flow, I*J));
      end

      return A;

end



function chernoff_objective(gamma::Float64,
                            policies::PolicyFunctions,
                            model::TwoCapitalEconomy,
                            grid::Grid_rz,
                            params::FinDiffMethod)

    G_findiff = Kolmogorov_FinDiff(policies, model, grid, params, gamma);
    return real(eigs(G_findiff, nev=1, which=:SM)[1][1]);
end



function finite_differences_2D(mat::Array{Float64, 2},
                               inner::Integer=5)

    # Finite differences in two dimensions
    mat_p1_0 = mat[inner + 2:end - inner + 1, inner + 1:end - inner]
    mat_m1_0 = mat[inner:end - inner - 1, inner + 1:end - inner]
    mat_0_0 = mat[inner + 1:end - inner, inner + 1:end - inner]

    mat_0_p1 = mat[inner + 1:end - inner, inner + 2:end - inner + 1]
    mat_0_m1 = mat[inner + 1:end - inner, inner:end - inner - 1]

    mat_p1_p1 = mat[inner + 2:end - inner + 1, inner + 2:end - inner + 1]
    mat_m1_m1 = mat[inner:end - inner - 1, inner:end - inner - 1]

    return mat_p1_0, mat_m1_0, mat_0_0, mat_0_p1, mat_0_m1, mat_p1_p1, mat_m1_m1
end



function change_of_variables(h::Array{Float64, 2},
                             g::Array{Float64, 2},
                             rr::Array{Float64, 2},
                             zz::Array{Float64, 2},
                             inner::Integer=5)

    I, J = size(zz);
    #dz = zz[1, 2] - zz[1, 1];

    h_p1_0, h_m1_0, h_0_0, h_0_p1, h_0_m1, h_p1_p1, h_m1_m1 = finite_differences_2D(h, inner);
    rr_p1_0, rr_m1_0, rr_0_0, rr_0_p1, rr_0_m1, rr_p1_p1, rr_m1_m1 = finite_differences_2D(rr, inner);
    zz_p1_0, zz_m1_0, zz_0_0, zz_0_p1, zz_0_m1, zz_p1_p1, zz_m1_m1 = finite_differences_2D(zz, inner);

    # Centered finite difference derivatives
    dh_r = (h_p1_0 - h_m1_0) ./ (rr_p1_0 - rr_m1_0);
    dh_z = (h_0_p1 - h_0_m1) ./ (zz_0_p1 - zz_0_m1);

    h_density = g[inner+1:end-inner, inner+1:end-inner] ./ sqrt.(dh_r.^2 + dh_z.^2);
    h_vec = reshape(h_0_0, (I-2*inner)*(J-2*inner))

    return h_vec, reshape(h_density, (I-2*inner)*(J-2*inner))
end



function change_of_variables2(h::Array{Float64, 2},
                             g::Array{Float64, 2},
                             rr::Array{Float64, 2},
                             zz::Array{Float64, 2},
                             inner::Integer=5)

    I, J = size(zz);
    #dz = zz[1, 2] - zz[1, 1];

    h_p1_0, h_m1_0, h_0_0, h_0_p1, h_0_m1, h_p1_p1, h_m1_m1 = finite_differences_2D(h, inner);
    rr_p1_0, rr_m1_0, rr_0_0, rr_0_p1, rr_0_m1, rr_p1_p1, rr_m1_m1 = finite_differences_2D(rr, inner);
    zz_p1_0, zz_m1_0, zz_0_0, zz_0_p1, zz_0_m1, zz_p1_p1, zz_m1_m1 = finite_differences_2D(zz, inner);

    # Centered finite difference derivatives
    dh_r = (h_p1_0 - h_m1_0) ./ (rr_p1_0 - rr_m1_0);
    dh_z = (h_0_p1 - h_0_m1) ./ (zz_0_p1 - zz_0_m1);

    h_density = g[inner+1:end-inner, inner+1:end-inner] ./ sqrt.(dh_r.^2 + dh_z.^2);
    h_vec = reshape(h_0_0, (I-2*inner)*(J-2*inner))

    return dh_r, dh_z
    #return h_vec, reshape(h_density, (I-2*inner)*(J-2*inner))
end




function consumption_dynamics(cons::Array{Float64, 2},
                              rr::Array{Float64, 2},
                              zz::Array{Float64, 2},
                              mu_1::Array{Float64, 2},
                              mu_r::Array{Float64, 2},
                              mu_z::Array{Float64, 2},
                              model::TwoCapitalEconomy,
                              inner::Integer=5)

    sigma_k1, sigma_k2 = model.k1.sigma_k, model.k2.sigma_k;
    sigma_z = model.k1.sigma_z
    pii = exp.(rr[inner + 1:end - inner, inner + 1:end - inner]) ./
               (1 .+ exp.(rr[inner + 1:end - inner, inner + 1:end - inner]));
    one_pii = 1 .- pii

    # Finite differences in two dimensions
    cons_p1_0, cons_m1_0, cons_0_0, cons_0_p1, cons_0_m1,
    cons_p1_p1, cons_m1_m1 = finite_differences_2D(cons, inner);
    rr_p1_0, rr_m1_0, rr_0_0, rr_0_p1, rr_0_m1,
    rr_p1_p1, rr_m1_m1 = finite_differences_2D(rr, inner);
    zz_p1_0, zz_m1_0, zz_0_0, zz_0_p1, zz_0_m1,
    zz_p1_p1, zz_m1_m1 = finite_differences_2D(zz, inner);


    dc_r = (cons_p1_0 - cons_m1_0) ./ (rr_p1_0 - rr_m1_0);
    dc_z = (cons_0_p1 - cons_0_m1) ./ (zz_0_p1 - zz_0_m1);

    dc_rr = (cons_p1_0 - 2*cons_0_0 + cons_m1_0) ./ (rr_p1_0 - rr_0_0).^2;
    dc_zz = (cons_0_p1 - 2*cons_0_0 + cons_0_m1) ./ (zz_0_p1 - zz_0_0).^2;
    dc_zr = (cons_p1_p1 - cons_p1_0 - cons_0_p1 +
                    2*cons_0_0 - cons_0_m1 - cons_m1_0 + cons_m1_m1) ./
                        (2 * (rr_p1_0 - rr_0_0) .* (zz_0_p1 - zz_0_0))

    logC_r = dc_r ./ cons_0_0;
    logC_z = dc_z ./ cons_0_0;
    sigma_r = (.01)*(sigma_k2 - sigma_k1);

    # log C = log K + log c^*(R, Z)
    logC_sigma = logC_r * sqrt(dot(sigma_r, sigma_r)) +
                 logC_z * sqrt(dot(sigma_z, sigma_z)) +
                 (.01)*sqrt.((one_pii .* sigma_k1[1] + pii .* sigma_k2[1]).^2 +
                             (one_pii .* sigma_k1[2] + pii .* sigma_k2[2]).^2 +
                             (one_pii .* sigma_k1[3] + pii .* sigma_k2[3]).^2);

    logC_mu = logC_r .* mu_r[inner+1:end-inner, inner+1:end-inner] +
              logC_z .* mu_z[inner+1:end-inner, inner+1:end-inner] +
                        mu_1[inner+1:end-inner, inner+1:end-inner] +
              (.5)*(  dc_rr*dot(sigma_r, sigma_r) +
                    2*dc_zr*dot(sigma_r, sigma_z) +
                      dc_zz*dot(sigma_z, sigma_z));

    return logC_mu, logC_sigma
end



function risk_free_rate(cons::Array{Float64, 2},
                        rr::Array{Float64, 2},
                        zz::Array{Float64, 2},
                        mu_1::Array{Float64, 2},
                        mu_r::Array{Float64, 2},
                        mu_z::Array{Float64, 2},
                        delta::Float64,
                        model::TwoCapitalEconomy, inner::Integer=5)


    sigma_k1, sigma_k2 = model.k1.sigma_k, model.k2.sigma_k;
    sigma_z = model.k1.sigma_z
    pii = exp.(rr[inner + 1:end - inner, inner + 1:end - inner]) ./
               (1 .+ exp.(rr[inner + 1:end - inner, inner + 1:end - inner]));
    one_pii = 1 .- pii

    # Finite differences in two dimensions
    cons_p1_0, cons_m1_0, cons_0_0, cons_0_p1, cons_0_m1,
    cons_p1_p1, cons_m1_m1 = finite_differences_2D(cons, inner);
    rr_p1_0, rr_m1_0, rr_0_0, rr_0_p1, rr_0_m1,
    rr_p1_p1, rr_m1_m1 = finite_differences_2D(rr, inner);
    zz_p1_0, zz_m1_0, zz_0_0, zz_0_p1, zz_0_m1,
    zz_p1_p1, zz_m1_m1 = finite_differences_2D(zz, inner);


    dc_r = (cons_p1_0 - cons_m1_0) ./ (rr_p1_0 - rr_m1_0);
    dc_z = (cons_0_p1 - cons_0_m1) ./ (zz_0_p1 - zz_0_m1);

    dc_rr = (cons_p1_0 - 2*cons_0_0 + cons_m1_0) ./ (rr_p1_0 - rr_0_0).^2;
    dc_zz = (cons_0_p1 - 2*cons_0_0 + cons_0_m1) ./ (zz_0_p1 - zz_0_0).^2;
    dc_zr = (cons_p1_p1 - cons_p1_0 - cons_0_p1 +
                    2*cons_0_0 - cons_0_m1 - cons_m1_0 + cons_m1_m1) ./
                        (2 * (rr_p1_0 - rr_0_0) .* (zz_0_p1 - zz_0_0))

    logC_r = dc_r ./ cons_0_0;
    logC_z = dc_z ./ cons_0_0;
    sigma_r = (.01)*(sigma_k2 - sigma_k1);

    logC_mu = logC_r .* mu_r[inner+1:end-inner, inner+1:end-inner] +
              logC_z .* mu_z[inner+1:end-inner, inner+1:end-inner] +
                        mu_1[inner+1:end-inner, inner+1:end-inner] +
              (.5)*(  dc_rr*dot(sigma_r, sigma_r) +
                    2*dc_zr*dot(sigma_r, sigma_z) +
                      dc_zz*dot(sigma_z, sigma_z));

    # log C = log K + log c^*(R, Z)
    logC_sigma_1 = logC_r .* sigma_r[1] + logC_z .* sigma_z[1] + (.01)*(one_pii .* sigma_k1[1] + pii .* sigma_k2[1]);
    logC_sigma_2 = logC_r .* sigma_r[2] + logC_z .* sigma_z[2] + (.01)*(one_pii .* sigma_k1[2] + pii .* sigma_k2[2]);
    logC_sigma_z = logC_r .* sigma_r[3] + logC_z .* sigma_z[3] + (.01)*(one_pii .* sigma_k1[3] + pii .* sigma_k2[3]);

    # Risk-free rate:
    r = (delta .+ logC_mu - (.5)*(logC_sigma_1.^2 + logC_sigma_2.^2 + logC_sigma_z.^2))

    return r
end





function impulse_init2(pii0::Float64,
                      z0::Float64,
                      dW0::Array{Float64, 1},
                      d1_interp::interpol,
                      d2_interp::interpol,
                      model::TwoCapitalEconomy,
                      grid::Grid_rz)

    #------------------------------------------------------------#
    # PARAMETERS
    #------------------------------------------------------------#
    I, J = grid.I, grid.J;
    alpha_z_hat = model.k1.alpha_z_hat;
    kappa_hat = model.k1.kappa_hat;

    alpha_k1_hat = model.k1.alpha_k_hat;
    alpha_k2_hat = model.k2.alpha_k_hat;
    beta1_hat = model.k1.beta_hat;
    beta2_hat = model.k2.beta_hat;

    s_k1 = model.k1.sigma_k;
    s_k2 = model.k2.sigma_k;
    s_z =  model.k1.sigma_z;

    phi1, phi2 = model.t1.phi, model.t2.phi;

    #------------------------------------------------------------#
    # INITIAL RESPONSES
    #------------------------------------------------------------#
    d1_0 = d1_interp(pii0, z0)
    d2_0 = d2_interp(pii0, z0)
    varphi1_0 = d1_0 - phi1*d1_0^2/2 + (.01)*(alpha_k1_hat + beta1_hat*z0)
    varphi2_0 = d2_0 - phi2*d2_0^2/2 + (.01)*(alpha_k2_hat + beta2_hat*z0)
    ito_adj = (.01)^2*((1-pii0)*dot(s_k1, s_k1) - pii0*dot(s_k2, s_k2) +
                       (2*pii0-1)*dot(s_k1, s_k2))

    dpii0 = pii0*(1-pii0)*((varphi2_0 - varphi1_0 + ito_adj) +
                                    dot((.01)*(s_k2 - s_k1), dW0))
    dz0 = alpha_z_hat - kappa_hat*z0 + dot(s_z, dW0)

    #------------------------------------------------------------#
    # VALUES AFTER IMPACT
    #------------------------------------------------------------#
    pii1 = pii0 + dpii0
    z1 = z0 + dz0
    r1 = log(pii1/(1-pii1))

    return pii1, z1, r1

end


function impulse(start::Array{Int64, 1},
                 dW0::Array{Float64, 1},
                 d1_interp::interpol,
                 d2_interp::interpol,
                 model::TwoCapitalEconomy,
                 grid::Grid_rz)

    I, J = grid.I, grid.J;
    r = range(grid.rmin, stop=grid.rmax, length=I);    # capital ratio vector
    z = range(grid.zmin, stop=grid.zmax, length=J);   # productivity vector

    ind_r0, ind_z0 = start
    r0, z0 = r[ind_r0], z[ind_z0]
    #------------------------------------------------------------#
    # PARAMETERS
    #------------------------------------------------------------#
    alpha_z_hat = model.k1.alpha_z_hat;
    kappa_hat = model.k1.kappa_hat;

    alpha_k1_hat = model.k1.alpha_k_hat;
    alpha_k2_hat = model.k2.alpha_k_hat;
    beta1_hat = model.k1.beta_hat;
    beta2_hat = model.k2.beta_hat;

    s_k1 = model.k1.sigma_k;
    s_k2 = model.k2.sigma_k;
    s_z =  model.k1.sigma_z;

    phi1, phi2 = model.t1.phi, model.t2.phi;

    #------------------------------------------------------------#
    # INITIAL RESPONSES
    #------------------------------------------------------------#
    d1_0 = d1_interp(r0, z0)
    d2_0 = d2_interp(r0, z0)
    varphi1_0 = d1_0 - phi1*d1_0^2/2 + (.01)*(alpha_k1_hat + beta1_hat*z0)
    varphi2_0 = d2_0 - phi2*d2_0^2/2 + (.01)*(alpha_k2_hat + beta2_hat*z0)
    ito_adj = (dot(s_k2, s_k2)-dot(s_k1, s_k1))*(.01)^2/2

    dr0 = varphi2_0 - varphi1_0 - ito_adj + dot((.01)*(s_k2 - s_k1), dW0)
    dz0 = alpha_z_hat - kappa_hat*z0 + dot(s_z, dW0)

    #------------------------------------------------------------#
    # VALUES AFTER IMPACT
    #------------------------------------------------------------#
    r1 = r0 + dr0
    z1 = z0 + dz0
    ind_r1 = argmin(abs.(r .- r1))
    ind_z1 = argmin(abs.(z .- z1))

    return [ind_r1, ind_z1]
end


function IRF(start::Array{Int64, 1},
             dW0::Array{Float64, 2},
             hor::Int64,
             P::SuiteSparse.UMFPACK.UmfpackLU{Float64,Int64},     #SparseMatrixCSC{Float64,Int64},
             P_wc::SuiteSparse.UMFPACK.UmfpackLU{Float64,Int64},  # SparseMatrixCSC{Float64,Int64},
             P_noR::SuiteSparse.UMFPACK.UmfpackLU{Float64,Int64}, #SparseMatrixCSC{Float64,Int64},
             pii::Array{Float64, 2},
             zz::Array{Float64, 2},
             d1::interpol,
             d2::interpol,
             model::TwoCapitalEconomy,
             policies::PolicyFunctions,
             grid::Grid_rz,
             label::Int64=1)

    I, J = grid.I, grid.J;
    impact_no = impulse(start, zeros(3), d1, d2, model, grid);
    impact_k = impulse(start, dW0[:, 2], d1, d2, model, grid);
    impact_z = impulse(start, dW0[:, 3], d1, d2, model, grid);
    pii_vec, zz_vec = vec(pii), vec(zz);

    h1 = -(policies.h1_F + policies.h1_B)/2;
    h2 = -(policies.h2_F + policies.h2_B)/2;
    hz = -(policies.hz_F + policies.hz_B)/2;
    h12_vec = vec((h1 + h2)/sqrt(2));
    hz_vec = vec(hz);

    # Period t distribution
    dist_t, dist_t_sK, dist_t_sZ = zeros(I, J), zeros(I, J), zeros(I, J);
    dist_t_price = zeros(I, J);
    dist_t[impact_no[1], impact_no[2]] = 1.;       # benchmark (no shock)
    dist_t_sK[impact_k[1], impact_k[2]] = 1.;      # impact of capital shocks
    dist_t_sZ[impact_z[1], impact_z[2]] = 1.;      # impact of Z shocks
    dist_t_price[start[1], start[2]] = 1.;

    # Storing matrices for outputs
    pii_path, z_path = zeros(hor, 3, 2), zeros(hor, 3, 2);
    shock_price_12, shock_price_z = zeros(hor), zeros(hor);
    pii_path[1, :, 1] .= sum(dist_t_sK .* pii) - sum(dist_t .* pii)
    pii_path[1, :, 2] .= sum(dist_t_sZ .* pii) - sum(dist_t .* pii)
    z_path[1, :, 1] .= sum(dist_t_sK .* zz) - sum(dist_t .* zz)
    z_path[1, :, 2] .= sum(dist_t_sZ .* zz) - sum(dist_t .* zz)
    shock_price_12[1] = sum(dist_t_price .* (h1 + h2)/sqrt(2));
    shock_price_z[1] = sum(dist_t_price .* hz)

    dist_t, dist_t_sK, dist_t_sZ = vec(dist_t), vec(dist_t_sK), vec(dist_t_sZ);
    dist_t_wc, dist_t_wc_sK, dist_t_wc_sZ = copy(dist_t), copy(dist_t_sK), copy(dist_t_sZ);
    dist_t_noR, dist_t_noR_sK, dist_t_noR_sZ = copy(dist_t), copy(dist_t_sK), copy(dist_t_sZ);
    dist_t_price = vec(dist_t_price);

    for t=2:hor
        # Baseline dynamics with robust decisions
        ldiv!(P, dist_t)
        ldiv!(P, dist_t_sK)
        ldiv!(P, dist_t_sZ)
        pii_path[t, 1, 1] = dot(dist_t_sK, pii_vec) - dot(dist_t, pii_vec)
        z_path[t, 1, 1] = dot(dist_t_sK, zz_vec) - dot(dist_t, zz_vec)
        pii_path[t, 1, 2] = dot(dist_t_sZ, pii_vec) - dot(dist_t, pii_vec)
        z_path[t, 1, 2] = dot(dist_t_sZ, zz_vec) - dot(dist_t, zz_vec)

        # Worst-case dynamics
        ldiv!(P_wc, dist_t_wc)
        ldiv!(P_wc, dist_t_wc_sK)
        ldiv!(P_wc, dist_t_wc_sZ)
        pii_path[t, 2, 1] = dot(dist_t_wc_sK, pii_vec) - dot(dist_t_wc, pii_vec)
        z_path[t, 2, 1] = dot(dist_t_wc_sK, zz_vec) - dot(dist_t_wc, zz_vec)
        pii_path[t, 2, 2] = dot(dist_t_wc_sZ, pii_vec) - dot(dist_t_wc, pii_vec)
        z_path[t, 2, 2] = dot(dist_t_wc_sZ, zz_vec) - dot(dist_t_wc, zz_vec)

        # Baseline model without robustness
        ldiv!(P_noR, dist_t_noR)
        ldiv!(P_noR, dist_t_noR_sK)
        ldiv!(P_noR, dist_t_noR_sZ)
        pii_path[t, 3, 1] = dot(dist_t_noR_sK, pii_vec) - dot(dist_t_noR, pii_vec)
        z_path[t, 3, 1] = dot(dist_t_noR_sK, zz_vec) - dot(dist_t_noR, zz_vec)
        pii_path[t, 3, 2] = dot(dist_t_noR_sZ, pii_vec) - dot(dist_t_noR, pii_vec)
        z_path[t, 3, 2] = dot(dist_t_noR_sZ, zz_vec) - dot(dist_t_noR, zz_vec)

        # Shock prices
        ldiv!(P_wc, dist_t_price)
        shock_price_12[t] = dot(dist_t_price, h12_vec);
        shock_price_z[t] = dot(dist_t_price, hz_vec)

    end

    println("Done with spec: ", label);

    return pii_path, z_path, shock_price_12, shock_price_z, label
end



function shock_price(start::Array{Int64, 1},
                     hor::Int64,
                     P_wc::SuiteSparse.UMFPACK.UmfpackLU{Float64,Int64},
                     policies::PolicyFunctions,
                     grid::Grid_rz)


    I, J = grid.I, grid.J;
    h1 = -(policies.h1_F + policies.h1_B)/2;
    h2 = -(policies.h2_F + policies.h2_B)/2;
    hz = -(policies.hz_F + policies.hz_B)/2;

    # Period t distribution
    dist_t = zeros(I, J);
    dist_t[start[1], start[2]] = 1.;

    # Storing matrices for outputs
    shock_price_12 = zeros(hor)
    shock_price_z = zeros(hor)
    shock_price_12[1] = sum(dist_t .* (h1 + h2)/sqrt(2));
    shock_price_z[1] = sum(dist_t .* hz)

    dist_t = vec(dist_t);
    h12_vec = vec((h1 + h2)/sqrt(2));
    hz_vec = vec(hz);

    for t=2:hor
        ldiv!(P_wc, dist_t)
        shock_price_12[t] = sum(dist_t .* h12_vec);
        shock_price_z[t] = sum(dist_t .* hz_vec)
    end

    return shock_price_12, shock_price_z
end

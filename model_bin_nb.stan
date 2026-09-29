// NB likelihood alone, for a fit with nothing assigned to the Normal component:
// the flat-prior posterior on (beta, log nu). The paper replaces these fits
// with the discrete construction (Supplement, Section 8).
//   r_i = exp(X_i' beta) t_i,  mu_i = r_i / nu  (Stan's NB2 with phi_i = r_i)

data {
  int<lower=1> n;                     // observations
  int<lower=1> r;                     // number of covariates (including intercept)
  matrix[n, r]  X;                    // design matrix  (n x r)
  vector<lower=0>[n] t;               // exposures
  array[n] int<lower=0> Y;            // integer counts for this bin  (n,)
}

parameters {
  vector[r] beta;                     // covariate effects for this bin
  real log_nu;                        // log dispersion; nu = exp(log_nu) > 0
}

transformed parameters {
  real<lower=0> nu = exp(log_nu);
}

model {
  vector[n] eta     = X * beta;
  vector[n] eta_c   = fmin(eta, rep_vector(20.0, n));
  vector[n] alpha_t = exp(eta_c) .* t;                              // phi for NB2
  vector[n] mu      = fmax(alpha_t / nu,  rep_vector(1e-6, n));
  vector[n] phi     = fmax(alpha_t,        rep_vector(1e-6, n));

  // Exact Negative Binomial likelihood for the discrete construction
  for (i in 1:n)
    target += neg_binomial_2_lpmf(Y[i] | mu[i], phi[i]);
}

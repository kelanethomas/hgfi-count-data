// Flat-prior posterior on (beta, log nu) for the hybrid model: the same
// likelihood as model_single_bin_hybrid.stan without the Jacobian.

data {
  int<lower=1> r;

  int<lower=0> n_norm;
  matrix[n_norm, r]  X_norm;
  vector<lower=0>[n_norm] t_norm;
  vector[n_norm] Y_norm;

  int<lower=0> n_nb;
  matrix[n_nb, r]  X_nb;
  vector<lower=0>[n_nb] t_nb;
  array[n_nb] int<lower=0> Y_nb;
}

parameters {
  vector[r] beta;
  real      log_nu;
}

transformed parameters {
  real<lower=0> nu = exp(log_nu);
}

model {
  // ---- Continuous observations: Normal likelihood (NO Jacobian) -----------
  if (n_norm > 0) {
    vector[n_norm] eta = fmin(X_norm * beta, rep_vector(20.0, n_norm));
    vector[n_norm] mu  = fmax(exp(eta) .* t_norm / nu, rep_vector(1e-6, n_norm));
    vector[n_norm] s2  = mu .* (nu + 1.0) / nu;
    target += normal_lpdf(Y_norm | mu, sqrt(s2));
  }

  // ---- Discrete observations: NB likelihood -------------------------------
  if (n_nb > 0) {
    vector[n_nb] eta = fmin(X_nb * beta, rep_vector(20.0, n_nb));
    vector[n_nb] at  = exp(eta) .* t_nb;
    vector[n_nb] mu  = fmax(at / nu, rep_vector(1e-6, n_nb));
    vector[n_nb] phi = fmax(at,      rep_vector(1e-6, n_nb));
    for (i in 1:n_nb)
      target += neg_binomial_2_lpmf(Y_nb[i] | mu[i], phi[i]);
  }
}

// Generalized fiducial density of the single-component hybrid model (Theorem 2):
// Normal likelihood and Jacobian for the Normal-assigned observations, NB
// likelihood alone for the rest.
//   r_i = exp(X_i' beta) t_i,  mu_i = r_i / nu,  sigma_i^2 = mu_i (1 + 1/nu)
// The Jacobian columns are the weights of Proposition 3; Stan samples log nu,
// so the nu column is multiplied by nu (the change of variables to log nu).

data {
  int<lower=1> r;                       // covariates (incl. intercept)

  int<lower=0> n_norm;                  // # continuous (Normal) observations
  matrix[n_norm, r]  X_norm;
  vector<lower=0>[n_norm] t_norm;
  vector[n_norm] Y_norm;

  int<lower=0> n_nb;                    // # discrete (NB) observations
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
  // ---- Continuous observations: Normal likelihood + GFD Jacobian ----------
  if (n_norm > 0) {
    vector[n_norm] eta = fmin(X_norm * beta, rep_vector(20.0, n_norm));
    vector[n_norm] mu  = fmax(exp(eta) .* t_norm / nu, rep_vector(1e-6, n_norm));
    vector[n_norm] s2  = mu .* (nu + 1.0) / nu;
    target += normal_lpdf(Y_norm | mu, sqrt(s2));

    vector[n_norm] w_beta = 0.5 * (Y_norm + mu);
    vector[n_norm] w_nu   = (-1.0 / (2.0 * nu * (nu + 1.0)))
                            .* (nu * mu + (nu + 2.0) * Y_norm);
    vector[n_norm] w_ln   = nu * w_nu;          // chain rule: d Y / d log_nu

    matrix[n_norm, r + 1] J   = append_col(diag_pre_multiply(w_beta, X_norm), w_ln);
    matrix[r + 1, r + 1]  JtJ = J' * J + diag_matrix(rep_vector(1e-8, r + 1));
    target += 0.5 * log_determinant(JtJ);
  }

  // ---- Discrete observations: NB likelihood only (no Jacobian) ------------
  if (n_nb > 0) {
    vector[n_nb] eta = fmin(X_nb * beta, rep_vector(20.0, n_nb));
    vector[n_nb] at  = exp(eta) .* t_nb;
    vector[n_nb] mu  = fmax(at / nu, rep_vector(1e-6, n_nb));
    vector[n_nb] phi = fmax(at,      rep_vector(1e-6, n_nb));
    for (i in 1:n_nb)
      target += neg_binomial_2_lpmf(Y_nb[i] | mu[i], phi[i]);
  }
}

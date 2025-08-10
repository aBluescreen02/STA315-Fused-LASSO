library(data.table)
library(genlasso)
library(glmnet)
library(Matrix)
library(MASS)

# SIMULATION SETTINGS #####

N <- 100                    # Number of samples
p <- 1000                   # Number of predictors
test_split_index <- floor(N * 0.8)  # Train-test split index

X.mean <- 0                 # Mean of predictors
X.sd <- 1                   # Standard deviation of predictors
X.correlation <- 0.2        # Correlation between predictors

beta.max_block_size <- 100  # Maximum size of beta non-zero blocks
beta.mean <- 0              # Mean of non-zero beta coefficients
beta.sd <- 1                # Standard deviation of non-zero beta coefficients

error.mean <- 0             # Mean of error terms
error.sd <- 20              # Standard deviation of error terms

fusedlasso_max_steps <- 250 # Max steps taken to solution path (increasing past 500 slows down performance a lot)
gamma.max <- 5              # Max gamma considered
gamma.interval <- 0.1       # Interval for gamma sequence
   
num_simulations <- 100      # Number of monte carlo simulations

# UTILITY FUNCTIONS #####

simulate_correlated_X <- function(N, p, rho) {
  # Create covariance matrix
  covariance_matrix <- toeplitz(rho^(0:(p - 1)))
  
  # Simulate multivariate normal rows
  X <- MASS::mvrnorm(n = N, mu = rep(0, p), Sigma = covariance_matrix)
  
  return(scale(X))  # standardize columns
}

generate_beta <- function(p, beta.max_block_size, beta.mean, beta.sd) {
  beta <- rep(0, p)
  
  num_blocks <- sample(1:(floor(p / beta.max_block_size)), 1)  # Number of non-zero blocks
  block_starts <- sort(sample(1:(p - beta.max_block_size), num_blocks)) # Start index of blocks
  
  for (start in block_starts) {
    len <- sample(1:beta.max_block_size, 1)
    end <- min(start + len, p)
    beta[start:end] <- rnorm(1, mean = beta.mean, sd = beta.sd)  # Assign values from a normal distribution
  }
  
  return(beta)
}

fit_lasso <- function(X_train, y_train, X_test, y_test) {
  cat("Fitting LASSO \n")
  
  cv_lasso <- cv.glmnet(X_train, y_train, alpha = 1)
  
  best_lambda <- NA
  best_error <- Inf
  best_beta <- rep(0, ncol(X_train))
  
  # Selects optimal lambda by c.v.
  for (lambda in cv_lasso$lambda) {
    lasso_model <- glmnet(X_train, y_train, alpha = 1, lambda = lambda)
    y_prediction <- predict(lasso_model, newx = X_test)
    test_error <- mean((y_test - y_prediction)^2)
    
    if (test_error < best_error) {
      best_lambda <- lambda
      best_error <- test_error
      best_beta <- as.vector(coefficients(lasso_model))
    }
  }
  
  return(list(error = best_error, lambda = best_lambda, beta = best_beta))
}

fit_fusedlasso <- function(X_train, y_train, X_test, y_test, gamma_seq = seq(gamma.interval,
                                                                             gamma.max, 
                                                                             length.out = (gamma.max / gamma.interval))) {
  cat("Fitting fusedlasso \n")
  
  best_gamma <- NA
  best_lambda <- NA
  best_error <- Inf
  best_beta <- rep(0, ncol(X_train))
  
  # Fits model for each gamma and iterates over lambda solution path
  for (gamma in gamma_seq) {
    if (which(gamma_seq == gamma) %% 25 == 0) {
      cat("Testing gamma:", gamma, "/", gamma.max, "\n")
    }
    
    model <- fusedlasso1d(y = y_train, 
                          X = X_train, 
                          gamma = gamma, 
                          verbose = FALSE,
                          maxsteps = fusedlasso_max_steps)
    
    for (lambda in model$lambda) {
      pred <- predict(model, lambda = lambda, Xnew = X_test)$fit
      error <- mean((y_test - pred)^2)
      
      if (error < best_error) {
        best_error <- error
        best_gamma <- gamma
        best_lambda <- lambda
        best_beta <- as.vector(coef.genlasso(model, lambda = lambda)$beta)
      }
    }
  }
  
  return(list(error = best_error, gamma = best_gamma, lambda = best_lambda, beta = best_beta))
}

# Why isn't this a default function to begin with?
`%nin%` <- function(x, table) !(x %in% table)

# RUN SIMULATION #####

set.seed(42)

X <- simulate_correlated_X(N, p, rho = X.correlation)

simulation_results <- list()

for (sim in 1:num_simulations) {
  cat("Running simulation:", sim, "\n")
  
  set.seed(42 + sim - 1)
  
  # Generate betas
  beta <- generate_beta(p, beta.max_block_size, beta.mean, beta.sd)
  
  # Generate errors
  errors <- rnorm(N, mean = error.mean, sd = error.sd)
  
  # Generate y's
  y <- X %*% beta + errors
  y <- y - mean(y)
  
  # Train-test split
  X_train <- X[1:test_split_index, ]
  y_train <- y[1:test_split_index]
  
  X_test <- X[(test_split_index + 1):N, ]
  y_test <- y[(test_split_index + 1):N]
  
  # Fit both models
  lasso_res <- fit_lasso(X_train, y_train, X_test, y_test)
  fused_res <- fit_fusedlasso(X_train, y_train, X_test, y_test)
  
  # Evaluate performance
  true_nonzero <- which(beta != 0)
  estimated_lasso <- which(abs(lasso_res$beta) > 1e-6)
  estimated_fused <- which(abs(fused_res$beta) > 1e-6)
  
  sensitivity_lasso <- mean(true_nonzero %in% estimated_lasso)
  specificity_lasso <- mean((1:p)[-true_nonzero] %nin% estimated_lasso)
  
  sensitivity_fused <- mean(true_nonzero %in% estimated_fused)
  specificity_fused <- mean((1:p)[-true_nonzero] %nin% estimated_fused)
  
  # Store results
  simulation_results[[sim]] <- list(
    lasso_error = lasso_res$error,
    fused_error = fused_res$error,
    lasso_lambda = lasso_res$lambda,
    fused_lambda = fused_res$lambda,
    fused_gamma = fused_res$gamma,
    sensitivity_lasso = sensitivity_lasso,
    specificity_lasso = specificity_lasso,
    sensitivity_fused = sensitivity_fused,
    specificity_fused = specificity_fused
  )
}

# FINAL STATS #####

results_df <- rbindlist(lapply(simulation_results, as.data.frame))

summary_stats <- results_df[, .(
  lasso_error = mean(lasso_error),
  fused_error = mean(fused_error),
  sensitivity_lasso = mean(sensitivity_lasso),
  specificity_lasso = mean(specificity_lasso),
  sensitivity_fused = mean(sensitivity_fused),
  specificity_fused = mean(specificity_fused)
)]

standard_errors <- results_df[, .(
  se_lasso_error = sd(lasso_error) / sqrt(.N),
  se_fused_error = sd(fused_error) / sqrt(.N),
  se_sensitivity_lasso = sd(sensitivity_lasso) / sqrt(.N),
  se_specificity_lasso = sd(specificity_lasso) / sqrt(.N),
  se_sensitivity_fused = sd(sensitivity_fused) / sqrt(.N),
  se_specificity_fused = sd(specificity_fused) / sqrt(.N)
)]

final_results <- cbind(summary_stats, standard_errors)

print(final_results)

cat("Average gamma across simulations:", mean(results_df$fused_gamma), "\n")

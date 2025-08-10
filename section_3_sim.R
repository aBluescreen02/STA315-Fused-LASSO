library(data.table)
library(ggplot2)
library(Matrix)
library(tidyverse)
library(graphics)
library(glmnet)
library(caret)
library(gridExtra)
library(nloptr)
library(osqp)
library(fusedlasso)
library(genlasso)

# PRESETS #####

set.seed(42)

N <- 20
p <- 100
test_split_index <- floor(N * 0.8)

sd <- 0.75
variance_error <- 1

# LINEAR MODEL #####

predictors_raw <- matrix(data = rnorm(N * p, mean = 0, sd = sd), nrow = N, ncol = p)
X <- scale(predictors_raw)

i <- 1:p

# for N = 20, p = 100
beta <- 5*(15 <= i & i <= 20) + 3.75*(50 <= i & i <= 65) + 4.5*(85 <= i & i <= 95)

# for N = 100, p = 1000
# beta <- 5*(150 <= i & i <= 155) + 3.75*(500 <= i & i <= 650) + 4.5*(850 <= i & i <= 950)

errors <- rnorm(N, mean = 0, sd = sqrt(variance_error))

y <- X %*% beta + errors
y <- y - mean(y)


# PREDICTIVE MODELS #####

univariate_coefs <- numeric(p)

for (j in 1:p) {
  univariate_model <- lm(y ~ X[, j])  
  univariate_coefs[j] <- coef(univariate_model)[2]
}

soft_threshold <- function(x, threshold) {
  sign(x) * max(0, abs(x) - threshold)
}

sigma <- sd(univariate_coefs)
N_j <- length(univariate_coefs)

#threshold <- sigma * sqrt(2 * log(N_j) / N_j)
threshold <- 10

soft_threshold_coefs <- sapply(univariate_coefs, soft_threshold, threshold)

# LASSO MODEL #####

X_train <- X[1:test_split_index, ]
y_train <- y[1:test_split_index]

X_test <- X[(test_split_index + 1):N, ]
y_test <- y[(test_split_index + 1):N]

cv_lasso <- cv.glmnet(X_train, y_train, alpha = 1)

#cv_lasso <- (1:100) * 0.1

best_lasso_lambda <- NA
best_lasso_error <- Inf

for (lambda in cv_lasso$lambda) {
  lasso_model <- glmnet(X_train, y_train, alpha = 1, lambda = lambda)
  y_prediction <- predict(lasso_model, newx = X_test)
  test_error <- mean((y_test - y_prediction)^2)
  
  if (test_error < best_lasso_error) {
    best_lasso_lambda <- lambda
    best_lasso_error <- test_error
  }
}

final_lasso <- glmnet(X_train, y_train, alpha = 1, lambda = best_lasso_lambda)
final_lasso_coefs <- as.matrix(coefficients(final_lasso))

# FUSION MODEL #####

#cv_fusion <- (1:100) * 0.1

fusion_model <- fusedlasso1d(y = y_train, X = X_train, verbose = FALSE, maxsteps = 1000)

best_fusion_lambda <- NA
best_fusion_error <- Inf

for (lambda in fusion_model$lambda) {
  
  y_prediction <- predict(fusion_model, lambda = lambda, Xnew = X_test)$fit
  test_error <- mean((y_test - y_prediction)^2)
  
  if (test_error < best_fusion_error) {
    best_fusion_lambda <- lambda
    best_fusion_error <- test_error
  }
}

final_fusion_coefs <- coef.genlasso(fusion_model, lambda = best_fusion_lambda)$beta

# FUSED LASSO MODEL #####

best_fusedlasso_lambda <- NA
best_fusedlasso_gamma <- Inf
best_fusedlasso_error <- Inf

for (gamma in (1:100) * 0.01) {
  fusedlasso_model <- fusedlasso1d(y = y_train, X = X_train, gamma = gamma, verbose = FALSE, maxsteps = 250)
  
  cat("Testing gamma:", gamma, "\n")
  
  for (lambda in fusedlasso_model$lambda) {
    y_prediction <- predict(fusedlasso_model, Xnew = X_test, lambda = lambda)$fit
    test_error <- mean((y_test - y_prediction)^2)
  
    if (test_error < best_fusedlasso_error) {
      best_fusedlasso_lambda <- lambda
      best_fusedlasso_gamma <- gamma
      
      best_fusedlasso_error <- test_error
    }
  }
}

final_fusedlasso_model <- fusedlasso1d(y = y_train, X = X_train, gamma = best_fusedlasso_gamma, maxsteps = )

final_fusedlasso_coefs <- coef.genlasso(final_fusedlasso_model, lambda = best_fusedlasso_lambda)$beta
  
# MODEL TO DATA FRAME #####

beta_df <- data.frame(
  index = seq_along(beta), 
  value = beta
)

univariate_df <- data.frame(
  index = seq_along(univariate_coefs),
  value = univariate_coefs
)

soft_threshold_df <- data.frame(
  index = seq_along(soft_threshold_coefs),
  value = soft_threshold_coefs
)

lasso_df <- data.frame(
  index = seq_along(final_lasso_coefs),
  value = as.vector(final_lasso_coefs)
)

fusion_df <- data.frame(
  index = seq_along(final_fusion_coefs),
  value = as.vector(final_fusion_coefs)
)

fusedlasso_df <- data.frame(
  index = seq_along(final_fusedlasso_coefs),
  value = as.vector(final_fusedlasso_coefs)
)

# GRAPHING #####

fig_3_a <- ggplot() + 
  geom_point(data = beta_df, aes(x = index, y = value), color = "black") +
  geom_line(data = beta_df, aes(x = index, y = value), color = "black") +
  
  geom_point(data = univariate_df, aes(x = index, y = value), color = "red") +
  geom_line(data = univariate_df, aes(x = index, y = value), color = "red") +
  
  geom_point(data = soft_threshold_df, aes(x = index, y = value), color = "green") +
  geom_line(data = soft_threshold_df, aes(x = index, y = value), color = "green") +
  
  labs(x = "Predictor", y = "Coefficient") +
  theme_minimal()

fig_3_b <- ggplot() + 
  geom_point(data = beta_df, aes(x = index, y = value), color = "black") +
  geom_line(data = beta_df, aes(x = index, y = value), color = "black") +
  
  geom_point(data = lasso_df, aes(x = index, y = value), color = "red") +
  geom_line(data = lasso_df, aes(x = index, y = value), color = "red") +
  
  labs(x = "Predictor", y = "Coefficient") +
  theme_minimal()

fig_3_c <- ggplot() + 
  geom_point(data = beta_df, aes(x = index, y = value), color = "black") +
  geom_line(data = beta_df, aes(x = index, y = value), color = "black") +
  
  geom_point(data = fusion_df, aes(x = index, y = value), color = "red") +
  geom_line(data = fusion_df, aes(x = index, y = value), color = "red") +
  
  labs(x = "Predictor", y = "Coefficient") +
  theme_minimal()

fig_3_d <- ggplot() + 
  geom_point(data = beta_df, aes(x = index, y = value), color = "black") +
  geom_line(data = beta_df, aes(x = index, y = value), color = "black") +
  
  geom_point(data = fusedlasso_df, aes(x = index, y = value), color = "red") +
  geom_line(data = fusedlasso_df, aes(x = index, y = value), color = "red") +
  
  labs(x = "Predictor", y = "Coefficient") +
  theme_minimal()

grid.arrange(fig_3_a, fig_3_b, fig_3_c, fig_3_d, ncol = 2, nrow = 2)

# TEMP TESTING #####

# Doesn't run

if(FALSE) {
  s_lasso <- 0.1  
  s_fusion <- 0.1  
  
  beta_pos <- pmax(beta, 0)
  beta_neg <- pmax(-beta, 0)
  
  L <- diag(p + 1)[-p, -1] * -1 + diag(p)
  # L <- -diag(p + 1)[-1, -(p + 1)] + diag(p)
  
  # old method to validate theta = L %*% beta 
  if (FALSE) {
    matrix_beta <- Matrix(beta, nrow = p)
    theta <- Matrix(0, nrow = p, ncol = 1)
    for (j in 1:length(new_beta)) {
      if (j == 1) {
        theta[j] = matrix_beta[j]
      } else {
        theta[j] = matrix_beta[j] - matrix_beta[j - 1]
      }
    }
    theta_pos <- pmax(theta, 0)
    theta_neg <- pmax(-theta, 0)
  }
  
  theta <- L %*% beta
  theta_pos <- pmax(theta, 0)
  theta_neg <- pmax(-theta, 0)
  
  e <- Matrix(1, nrow = p, ncol = 1)
  e_0 <- e
  e_0[1, 1] = 0
  
  a_0 <- Matrix(0, nrow = p * 2 - 1, ncol = 1)
  a_0[1, 1] = Inf
  
  I_p <- diag(p)
  
  # minimize <- Matrix(t(y - X %*% beta) %*% s * (y - X %*% beta))
  
  P <- Matrix(2 * t(X) %*% X)
  q <- Matrix(-2 * t(X) %*% y)
  
  A <- rbind(
    cbind(L, Matrix(0, nrow = p, ncol = 2 * p), -I_p, I_p),  # Theta = L * beta
    cbind(I_p, -I_p, I_p, Matrix(0, nrow = p, ncol = 2 * p)),  # Beta = beta_pos - beta_neg
    cbind(Matrix(0, nrow = 1, ncol = p), t(e), t(e), Matrix(0, nrow = 1, ncol = 2 * p)),  # Sparsity constraint
    cbind(Matrix(0, nrow = 1, ncol = 3 * p), t(e_0), t(e_0))   # Fusion constraint
  )
  
  num_vars <- 5 * p
  x <- Matrix(0, nrow = num_vars, ncol = 1)
  
  #lower <- c(-a_0, 0., 0., 0.)
  #upper <- c(a_0, 0, s_lasso, s_fusion)
  
  lower <- c(rep(0., p * 2 + 2))
  lower[1] <- -Inf
  
  upper <- c(rep(0., p * 2 + 2))
  upper[1] <- Inf
  upper[(p * 2 + 1):(p * 2 + 2)] = c(s_lasso, s_fusion)
  
  # lower <- rbind(-a_0, 0, 0, 0)
  # upper <- rbind(a_0, 0, s_lasso, s_fusion)
  
  model <- osqp(P, q, A, lower, upper)
  result <- model$Solve()
  
  # sqp attempt
  
  s_lasso <- 10
  s_fusion <- 10
  
  L <- L <- diag(p + 1)[-p, -1] * -1 + diag(p)
  
  S <- s_lasso * Diagonal(p) + s_fusion * crossprod(L)
  
  P <- Matrix(2 * t(X) %*% X)
  q <- Matrix(-2 * t(X) %*% y)
  
  objective <- function(beta) {
    return(as.numeric(t(beta) %*% P %*% beta + t(q) %*% beta))
  }
  
  gradient <- function(beta) {
    return(as.numeric(2 * P %*% beta + q))
  }
  
  constraint_lasso <- function(beta) {
    return(sum(abs(beta)) - s_lasso)
  }
  
  constraint_fusion <- function(beta) {
    return(sum(abs(diff(beta))) - s_fusion)
  }
  
  result <- slsqp(
    x0 = rep(0, p),
    fn = objective,
    gr = gradient,
    hin = function(beta) c(constraint_lasso(beta), constraint_fusion(beta)),
    nl.info = TRUE,
    control = list(xtol_rel = 1e-6)
  )
  
  fused_lasso_coefs <- result$par
  
  # fusedlasso attempt
  
  max_lambda <- 10
  lambda_interval <- 0.1
  
  error_grid <- Matrix(0, nrow = max_lambda / lambda_interval, ncol = max_lambda / lambda_interval)
  
  predict_fusedlasso <- function(model, X_new) {
    return(X_new %*% model$beta + model$intercept)
  }
  
  for(s_lasso in 1:ncol(lambda_grid)) {
    
    cat("testing:", s_lasso, "\n")
    
    for(s_fusion in 1:nrow(lambda_grid)) {
      res <- fusedlasso(X_train, y_train,
                        lambda.lasso = s_lasso * lambda_interval, 
                        lambda.fused = s_fusion * lambda_interval, 
                        family = "gaussian")
      
      y_prediction <- predict_fusedlasso(res, X_test)
      
      test_error <- mean((y_test - y_prediction)^2)
      
      error_grid[s_lasso, s_fusion] <- test_error
    }
  }
  
  best_fusedlasso_error <- min(error_grid)
  best_fusedlasso_lambda <- which(error_grid == best_fusedlasso_error, arr.ind = TRUE)
  
  final_fusedlasso <- fusedlasso(X,
                                 y, 
                                 lambda.lasso = best_fusedlasso_lambda[1] * lambda_interval, 
                                 lambda.fused = best_fusedlasso_lambda[2] * lambda_interval, 
                                 family = "gaussian")
  fusedlasso_coefs <- final_fusedlasso$beta
  
  # cv fusion (works?)
  
  fusion_model <- fusedlasso1d(y, X = X)
  
  n_folds <- 10
  folds <- sample(rep(1:n_folds, length.out = N))
  
  cv_errors <- rep(0, length(fusion_model$lambda))
  
  for (j in 1:n_folds) {
    train_idx <- which(folds != 1)
    test_idx <- which(folds == 1)
    
    X_train <- X[train_idx, ]
    X_test <- X[test_idx, ]
    
    y_train <- y[train_idx]
    y_test <- y[test_idx]
    
    fit_cv <- fusedlasso1d(y = y_train, X = X_train)
    
    predictions <- predict.genlasso(fit_cv, Xnew = X_test)$fit
    
    mse <- colMeans((predictions - y_test)^2)
    
    cv_errors <- cv_errors + mse
  }
  
  cv_errors <- cv_errors / n_folds
  
  best_fusion_lambda <- fusion_model$lambda[which.min(cv_errors)]
  
  final_fusion_coefs <- coef.genlasso(fusion_model, lambda = best_fusion_lambda)$beta
  
  # fusedlasso1d example 
  
  set.seed(1)
  n = 100
  i = 1:n
  y_temp = (i > 20 & i < 30) + 5*(i > 50 & i < 70) + rnorm(n, sd=0.1)
  out = fusedlasso1d(y_temp)
  beta1 = coef(out, lambda=1.5)$beta
}


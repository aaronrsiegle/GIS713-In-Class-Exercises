# =============================================================================
# GIS 713  ML III  -  Demo 4: Neural network basics with nnet
# =============================================================================
#   Part 1  One neuron (no hidden layer) is logistic regression
#   Part 2  A nonlinear habitat: why hidden units help
#   Part 3  Tune the two key settings: size and decay
#
# Data are simulated, so we know the right answer.
# Package: nnet (comes with R)
# =============================================================================

library(nnet)
set.seed(713)

accuracy <- function(observed, predicted) mean(observed == predicted)


# =============================================================================
# Part 1  One neuron (no hidden layer) is logistic regression
# =============================================================================
# Wetland presence (yes/no) depends on a wetness index and distance to road.

n <- 400
wet_data <- data.frame(wetness   = rnorm(n),
                       dist_road = runif(n, 0, 2))      # km
p_true <- plogis(-0.5 + 1.8 * wet_data$wetness - 1.0 * wet_data$dist_road)
wet_data$wetland <- factor(rbinom(n, 1, p_true), labels = c("no", "yes"))

# (a) Ordinary logistic regression
glm_fit <- glm(wetland ~ wetness + dist_road, data = wet_data, family = binomial)

# (b) A "network" with no hidden layer:
#     size = 0     -> no hidden units
#     skip = TRUE  -> connect the inputs straight to the output neuron
# If you have 0 hidden nodes then it just logisitic regression
nn_fit <- nnet(wetland ~ wetness + dist_road, data = wet_data,
               size = 0, skip = TRUE, trace = FALSE)

# Compare the coefficients: (Intercept) = bias "b->o", slopes = weights "i->o"
cbind(glm = coef(glm_fit), one_neuron = coef(nn_fit))
# Same numbers. A neural network is built from logistic regressions.


# =============================================================================
# Part 2  A nonlinear habitat: why hidden units help
# =============================================================================
# A salamander lives at MID elevations (220-380 m) and near streams.
# A single neuron can only draw one straight boundary, so it can't capture "mid".

make_habitat <- function(n) {
  d <- data.frame(elev        = runif(n, 100, 500),     # m
                  dist_stream = runif(n, 0, 1000))      # m
  suitable <- 2.5 * (d$elev > 220 & d$elev < 380) - 0.004 * d$dist_stream
  d$present <- factor(rbinom(n, 1, plogis(3 * (suitable - 0.5))),
                      labels = c("absent", "present"))
  d
}
train <- make_habitat(400)
test  <- make_habitat(5000)          # a big independent test set

# --- Step 1: scale the predictors (mean 0, sd 1) ---------------------------
# Neural networks need this; trees don't. Use the TRAINING means and sds for
# both data sets, so no information from the test set leaks into training.
# raw values have vastly different ranges 
predictors <- c("elev", "dist_stream")
means <- sapply(train[predictors], mean)
sds   <- sapply(train[predictors], sd)
scale_with <- function(d) {
  d[predictors] <- scale(d[predictors], center = means, scale = sds)
  d
}
train_s <- scale_with(train)
test_s  <- scale_with(test)

# --- Step 2: compare networks of different sizes ---------------------------
# no hidden layer (= logistic regression)
m0 <- nnet(present ~ elev + dist_stream, data = train_s,
           size = 0, skip = TRUE, trace = FALSE)
# one hidden layer with 3 units, a little weight decay
# maxit = maximum interations 
m3 <- nnet(present ~ elev + dist_stream, data = train_s,
           size = 3, decay = 0.01, maxit = 1000, trace = FALSE)
# one hidden layer with 30 units, no weight decay (prone to overfitting)
m30 <- nnet(present ~ elev + dist_stream, data = train_s,
            size = 30, decay = 0, maxit = 1000, trace = FALSE)

results <- data.frame(
  model    = c("no hidden layer", "3 hidden, decay 0.01", "30 hidden, decay 0"),
  train_acc = c(accuracy(train_s$present, predict(m0,  train_s, type = "class")),
                accuracy(train_s$present, predict(m3,  train_s, type = "class")),
                accuracy(train_s$present, predict(m30, train_s, type = "class"))),
  test_acc  = c(accuracy(test_s$present,  predict(m0,  test_s,  type = "class")),
                accuracy(test_s$present,  predict(m3,  test_s,  type = "class")),
                accuracy(test_s$present,  predict(m30, test_s,  type = "class"))))
results
# The 30-unit net is nearly perfect on the training data but no better on new
# data: that gap is overfitting.
# Model 3 has an overfitting issue 
# What did the 3-unit network learn? Predict over a grid of elevations,
# holding distance to stream at 100 m, and plot (on the original scale).
grid <- data.frame(elev = 100:500, dist_stream = 100)
grid$p_m0 <- predict(m0, scale_with(grid), type = "raw")
grid$p_m3 <- predict(m3, scale_with(grid), type = "raw")
plot(grid$elev, grid$p_m3, type = "l", lwd = 2, ylim = c(0, 1),
     xlab = "Elevation (m)", ylab = "P(present), 100 m from a stream")
lines(grid$elev, grid$p_m0, lwd = 2, lty = 2)
legend("topright", c("3 hidden units", "no hidden layer"), lty = 1:2, lwd = 2)

# Try it: refit m3 on the UNscaled data (train instead of train_s) and compare
# its test accuracy.


# =============================================================================
# Part 3  Tune size and decay with cross-validation
# =============================================================================
#   size  = number of hidden units (flexibility)
#   decay = penalty on large weights (smoothness; protects against overfitting)
# These points are independent random locations, so ordinary random folds are
# fine here. (Demo 5 shows why spatial data need spatial folds.)

# 5-fold CV, repeated 3 times with different random fold assignments.
# (One 5-fold split is noisy; repeating it gives a steadier answer.)
fold_sets <- lapply(1:3, function(r) sample(rep(1:5, length.out = nrow(train))))

# Returns the accuracy in each held-out fold (3 repeats x 5 folds = 15 values)
cv_accuracy <- function(size, decay) {
  fold_acc <- c()
  for (folds in fold_sets) for (k in 1:5) {
    fit_rows  <- train[folds != k, ]
    test_rows <- train[folds == k, ]
    # scale using the fitting rows only
    mu <- sapply(fit_rows[predictors], mean)
    s  <- sapply(fit_rows[predictors], sd)
    fit_rows[predictors]  <- scale(fit_rows[predictors],  center = mu, scale = s)
    test_rows[predictors] <- scale(test_rows[predictors], center = mu, scale = s)
    m <- nnet(present ~ elev + dist_stream, data = fit_rows, size = size,
              decay = decay, maxit = 1000, trace = FALSE)
    # computing the accuracy 
    fold_acc <- c(fold_acc, accuracy(test_rows$present, predict(m, test_rows, type = "class")))
  }
  fold_acc
}

# We need to create a grid to find the optimal combination of the two parameters 
# gives you 20 different hyperparameter combos that we need to fit 
tuning <- expand.grid(size = c(1, 2, 4, 8, 16), decay = c(0, 0.01, 0.1, 0.5))
fold_results <- mapply(cv_accuracy, tuning$size, tuning$decay)   # 15 rows, one column per setting
tuning$cv_acc <- colMeans(fold_results)
tuning$se     <- apply(fold_results, 2, sd) / sqrt(15)             # standard error
tuning[order(-tuning$cv_acc), ] # reordering them based on cv accuracy 

# Many settings score almost the same. One-standard-error rule: every setting
# within 1 SE of the best is "as good, within noise"; of those, pick the most
# regularized one (largest decay, then fewest hidden units).
# hyperparameter tuning by hand 
# packages like caret can automate this 
top     <- tuning[which.max(tuning$cv_acc), ]
as_good <- tuning[tuning$cv_acc >= top$cv_acc - top$se, ]
as_good
best <- as_good[order(-as_good$decay, as_good$size), ][1, ]
best

# Refit the chosen settings on ALL training data, then check on the test set.
# Networks start from random weights, so fit 5 and average their probabilities.
final_nets <- lapply(1:5, function(i)
  nnet(present ~ elev + dist_stream, data = train_s, size = best$size,
       decay = best$decay, maxit = 1000, trace = FALSE))
p_avg <- rowMeans(sapply(final_nets, predict, newdata = test_s, type = "raw"))
accuracy(test_s$present, ifelse(p_avg > 0.5, "present", "absent"))

# Questions:
#  * Which settings do badly, and why (look at size 1 and at decay 0)?
#  * Rerun Part 3 with a different seed. Does the top-scoring setting change?
#    Does the 1-SE choice change as much?
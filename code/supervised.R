################################################################################
# STEP 4: SUPERVISED
################################################################################

# Remove redundant variables
drop_cols <- c("profit", "state", "city", "sub_category")

model_df <- data %>%
  mutate(
    log_sales = log1p(sales)
  ) %>%
  select(-any_of(drop_cols), -sales)

# Baseline setting for categoric variables
most_frequent <- function(x) names(sort(table(x), decreasing = TRUE))[1]

factor_vars <- c("region", "ship_mode", "category", "segment")

cat("\n Chosen baseline (= most frequent category):\n")
for (v in factor_vars) {
  ref <- most_frequent(model_df[[v]])
  model_df[[v]] <- relevel(model_df[[v]], ref = ref)
  cat(" ", v, ":", ref, "\n")
}

# =================================================================================
# Train / Test Split (70/30)

set.seed(42)
train_index <- createDataPartition(model_df$loss_flag,
                                   p    = 0.70,
                                   list = FALSE)

train_df <- model_df[train_index, ]
test_df  <- model_df[-train_index, ]

# =================================================================================
# Helper function: compute confusion matrix for final evaluation
eval_model <- function(pred, actual, name) {
  cm <- table(Predicted = pred, Actual = actual)
  TN <- cm[1,1]; FP <- cm[2,1]; FN <- cm[1,2]; TP <- cm[2,2]
  prec <- TP / (TP + FP)
  rec  <- TP / (TP + FN)
  data.frame(
    Model       = name,
    Accuracy    = round((TP + TN) / sum(cm), 4),
    Precision   = round(prec, 4),
    Recall      = round(rec, 4),
    Specificity = round(TN / (TN + FP), 4),
    F1          = round(2 * prec * rec / (prec + rec), 4)
  )
}

results <- list()

# =================================================================================
# glmnet/xgboost need numeric matrix, not data.frame with factor

formula_pen <- loss_flag ~ .
X_train <- model.matrix(formula_pen, data = train_df)[, -1]
X_test  <- model.matrix(formula_pen, data = test_df)[, -1]

y_train <- as.numeric(as.character(train_df$loss_flag))
y_test  <- as.numeric(as.character(test_df$loss_flag))

# =================================================================================
# LOGISTIC REGRESSION
# =================================================================================

logit_model <- glm(loss_flag ~ .,
                   data = train_df, family = binomial(link = "logit"))

cat("\n========== MODEL SUMMARY ==========\n")
summary(logit_model)

pred_logit <- ifelse(predict(logit_model, newdata = test_df, type = "response") >= 0.5, 1, 0)
results[["Logistic"]] <- eval_model(pred_logit, test_df$loss_flag, "Logistic")

# =================================================================================
# LASSO
# =================================================================================

set.seed(42)
cv_lasso <- cv.glmnet(X_train, y_train,
                      family = "binomial",
                      alpha  = 1,          # Lasso (L1 penalty)
                      nfolds = 10,
                      type.measure = "class")

plot(cv_lasso)
title("LASSO — Cross-Validation", line = 3)

# Coefficients and feature selection with lambda.1se
lasso_coefs <- coef(cv_lasso, s = "lambda.1se")
cat("\n--- LASSO: Coefficienti (lambda.1se) ---\n")
print(lasso_coefs)

pred_lasso <- ifelse(predict(cv_lasso, newx = X_test, s = "lambda.1se", type = "response") >= 0.5, 1, 0)
results[["LASSO"]] <- eval_model(pred_lasso, y_test, "LASSO")

# =================================================================================
# 6.5  RIDGE
# =================================================================================

set.seed(42)
cv_ridge <- cv.glmnet(X_train, y_train,
                      family = "binomial",
                      alpha  = 0,          # Ridge
                      nfolds = 10,
                      type.measure = "class")

# -log(lambda) - MSE plot
plot(cv_ridge)
title("RIDGE — Cross-Validation", line = 3)

# Coefficients with lambda.1se
ridge_coefs <- coef(cv_ridge, s = "lambda.1se")
cat("\n--- RIDGE: CoefficientS (lambda.1se) ---\n")
print(ridge_coefs)

pred_ridge <- ifelse(predict(cv_ridge, newx = X_test, s = "lambda.1se", type = "response") >= 0.5, 1, 0)
results[["RIDGE"]] <- eval_model(pred_ridge, y_test, "RIDGE")

# =================================================================================
# RANDOM FOREST
# =================================================================================

set.seed(42)
p <- ncol(train_df) - 1
mtry_values <- 2:p

# best number of variables to TRY at each split
oob_errors <- sapply(mtry_values, function(m) {
  randomForest(loss_flag ~ ., data = train_df,
               ntree = 500, mtry = m, nodesize = 5)$err.rate[500, "OOB"]
})
best_mtry <- mtry_values[which.min(oob_errors)]

# tuning mtry plot
plot(mtry_values, oob_errors, type = "b", pch = 19,
     xlab = "mtry", ylab = "OOB Error Rate",
     main = "Tuning mtry — Random Forest")

set.seed(42)
rf_model <- randomForest(loss_flag ~ ., data = train_df,
                         ntree = 500, mtry = best_mtry,
                         nodesize = 5, importance = TRUE)
print(rf_model)

# Variable importance plot 
imp_df <- as.data.frame(importance(rf_model)) %>%
  mutate(variable = rownames(.)) %>%
  pivot_longer(cols = c("MeanDecreaseAccuracy", "MeanDecreaseGini"),
               names_to = "metric", values_to = "value")
ggplot(imp_df, aes(x = value, y = reorder(variable, value), fill = metric)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~metric, scales = "free_x",
             labeller = labeller(metric = c(
               MeanDecreaseAccuracy = "Mean Decrease Accuracy",
               MeanDecreaseGini = "Mean Decrease Gini"))) +
  scale_fill_manual(values = c("MeanDecreaseAccuracy" = "#2C7BB6",
                               "MeanDecreaseGini" = "#D7191C")) +
  labs(title = "Random Forest — Variable Importance",
       x = NULL, y = NULL) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5, size = 15),
    strip.text = element_text(face = "bold", size = 12),
    panel.grid.major.y = element_blank(),
    axis.text.y = element_text(size = 11)
  )

pred_rf <- predict(rf_model, newdata = test_df, type = "class")
results[["Random Forest"]] <- eval_model(pred_rf, test_df$loss_flag, "Random Forest")

# =================================================================================
# BOOSTING
# =================================================================================

# =================================================================================
# GBM 

train_gbm <- as.data.frame(X_train)
train_gbm$loss_flag <- y_train

test_gbm <- as.data.frame(X_test)
test_gbm$loss_flag <- y_test

set.seed(42)
gbm_model <- gbm(loss_flag ~ .,
                  data         = train_gbm,
                  distribution = "bernoulli",
                  n.trees      = 3000,
                  interaction.depth = 4,
                  shrinkage    = 0.01,
                  n.minobsinnode = 10,
                  cv.folds     = 10,
                  verbose      = FALSE)

# Optimal number of trees
best_n_trees_gbm <- gbm.perf(gbm_model, method = "cv")
cat("\n--- GBM: Optimal number of trees ---\n")
cat("n.optimal trees:", best_n_trees_gbm, "\n")

# Variable Importance Plot
gbm_imp <- summary(gbm_model, n.trees = best_n_trees_gbm, plotit = FALSE)
gbm_imp$var <- factor(gbm_imp$var, levels = rev(gbm_imp$var))

ggplot(gbm_imp, aes(x = rel.inf, y = var)) +
  geom_col(fill = "steelblue") +
  geom_text(aes(label = round(rel.inf, 1)), hjust = -0.2, size = 3.2) +
  labs(title = "GBM — Variable Importance",
       x = "Relative Influence (%)", y = NULL) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

pred_gbm <- ifelse(predict(gbm_model, newdata = test_gbm,
                          n.trees = best_n_trees_gbm, type = "response") >= 0.5, 1, 0)
results[["GBM"]] <- eval_model(pred_gbm, test_gbm$loss_flag, "GBM")

# =================================================================================
# XGBoost

dtrain <- xgb.DMatrix(data = X_train, label = y_train)
dtest  <- xgb.DMatrix(data = X_test,  label = y_test)

params_xgb <- list(
  objective        = "binary:logistic",
  eval_metric      = "error",
  eta              = 0.01,
  max_depth        = 4,
  min_child_weight = 1,
  subsample        = 0.8,
  colsample_bytree = 0.8,
  scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)
)

set.seed(42)
cv_xgb <- xgb.cv(params  = params_xgb,
                  data    = dtrain,
                  nrounds = 3000,
                  nfold   = 10,
                  early_stopping_rounds = 50,
                  print_every_n = 100,
                  verbose = TRUE)

best_nrounds_xgb <- cv_xgb$early_stop$best_iteration
cat("\n--- XGBoost: Optimal nrounds (CV) ---\n")
cat("best_iteration:", best_nrounds_xgb, "\n")

set.seed(42)
xgb_model <- xgb.train(params  = params_xgb,
                        data    = dtrain,
                        nrounds = best_nrounds_xgb,
                        verbose = FALSE)

xgb_imp <- xgb.importance(model = xgb_model)

# Variable importance plot - XGBoost
xgb_imp_df <- as.data.frame(xgb_imp) %>%
  arrange(desc(Gain)) %>%
  slice_head(n = min(20, nrow(xgb_imp)))

ggplot(xgb_imp_df, aes(x = Gain, y = reorder(Feature, Gain))) +
  geom_col(fill = "#2C7BB6") +
  geom_text(aes(label = round(Gain, 3)), hjust = -0.1, size = 3.5) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.15))) +
  labs(title = "XGBoost — Variable Importance",
       x = "Gain", y = NULL) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title    = element_text(face = "bold", hjust = 0.5, size = 15),
    plot.subtitle = element_text(hjust = 0.5, color = "gray50"),
    panel.grid.major.y = element_blank(),
    axis.text.y   = element_text(size = 11)
  )

# PDP - discount - XGBoost
pd <- partial(xgb_model,
              pred.var = "discount",
              train    = X_train,
              plot     = FALSE,
              prob     = TRUE)

ggplot(pd, aes(x = discount, y = yhat)) +
  geom_line(color = "#2C7BB6", linewidth = 1.2) +
  geom_ribbon(aes(ymin = min(yhat), ymax = yhat), fill = "#2C7BB6", alpha = 0.1) +
  labs(title    = "Partial Dependence Plot (XGBoost) — discount",
       x = "discount",
       y = "Predicted Probability (loss_flag = 1)") +
  theme_minimal(base_size = 13) +
  theme(
    plot.title    = element_text(face = "bold", hjust = 0.5, size = 15),
    plot.subtitle = element_text(hjust = 0.5, color = "gray50")
  )

pred_xgb <- ifelse(predict(xgb_model, newdata = dtest) >= 0.5, 1, 0)
results[["XGBoost"]] <- eval_model(pred_xgb, y_test, "XGBoost")

# =================================================================================
# CONFRONTO FINALE MODELLI
# =================================================================================

comparison <- do.call(rbind, results)
rownames(comparison) <- NULL

cat("\n========== METRICS COMPARISON — ALL MODELS ==========\n")
print(comparison)

# Confusion matrix All models (precision - recall)
comparison_long <- comparison %>%
  pivot_longer(cols = c(Precision, Recall),
               names_to = "metric", values_to = "value")

ggplot(comparison_long, aes(x = reorder(Model, -value), y = value, fill = metric)) +
  geom_col(position = "dodge") +
  geom_text(aes(label = round(value, 3)),
            position = position_dodge(width = 0.9),
            vjust = -0.3, size = 3.2) +
  scale_fill_manual(values = c("Precision" = "#2C7BB6", "Recall" = "#D7191C")) +
  labs(title    = "Precision and Recall Comparison — All Models",
       subtitle = "Positive class: loss_flag = 1",
       x = NULL, y = NULL, fill = NULL) +
  ylim(0, 1) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title    = element_text(face = "bold", hjust = 0.5, size = 15),
    plot.subtitle = element_text(hjust = 0.5, color = "gray50"),
    legend.position = "top"
  )


# ROC curve (AUC)
# Predicted probabilities — all models
prob_logit <- predict(logit_model, newdata = test_df, type = "response")
prob_lasso <- as.vector(predict(cv_lasso, newx = X_test, s = "lambda.1se", type = "response"))
prob_ridge <- as.vector(predict(cv_ridge, newx = X_test, s = "lambda.1se", type = "response"))
prob_rf <- predict(rf_model, newdata = test_df, type = "prob")[, "1"]
prob_gbm <- predict(gbm_model, newdata = test_gbm, n.trees = best_n_trees_gbm, type = "response")
prob_xgb <- predict(xgb_model, newdata = dtest)

# ROC objects
roc_logit <- roc(as.numeric(as.character(test_df$loss_flag)), prob_logit, quiet = TRUE)
roc_lasso  <- roc(y_test, prob_lasso,  quiet = TRUE)
roc_ridge  <- roc(y_test, prob_ridge,  quiet = TRUE)
roc_rf     <- roc(y_test, prob_rf,     quiet = TRUE)
roc_gbm    <- roc(y_test, prob_gbm,    quiet = TRUE)
roc_xgb    <- roc(y_test, prob_xgb,    quiet = TRUE)

# Build unified dataframe
make_roc_df <- function(roc_obj, name) {
  data.frame(
    FPR   = 1 - roc_obj$specificities,
    TPR   = roc_obj$sensitivities,
    Model = paste0(name, "  (AUC = ", round(auc(roc_obj), 4), ")")
  )
}

roc_df <- rbind(
  make_roc_df(roc_logit, "Logistic"),
  make_roc_df(roc_lasso,  "LASSO"),
  make_roc_df(roc_ridge,  "Ridge"),
  make_roc_df(roc_rf,     "Random Forest"),
  make_roc_df(roc_gbm,    "GBM"),
  make_roc_df(roc_xgb,    "XGBoost")
)

# Dynamic color palette
palette_roc <- setNames(
  c("#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#A65628", "#E41A1C"),
  unique(roc_df$Model)
)

ggplot(roc_df, aes(x = FPR, y = TPR, color = Model)) +
  geom_line(linewidth = 0.9) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey60") +
  scale_color_manual(values = palette_roc) +
  scale_x_continuous(labels = percent_format(accuracy = 1), limits = c(0, 1)) +
  scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, 1)) +
  labs(
    title    = "ROC Curve — All Models Comparison",
    subtitle = "Test set (stratified 70/30 split)",
    x        = "False Positive Rate (1 − Specificity)",
    y        = "True Positive Rate (Recall)",
    color    = NULL
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title        = element_text(face = "bold"),
    legend.position   = "bottom",
    legend.background = element_rect(fill = "white", color = "grey80"),
    legend.text       = element_text(size = 9)
  ) +
  guides(color = guide_legend(ncol = 2))





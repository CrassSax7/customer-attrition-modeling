# ===========================================================
# Modeling Customer Attrition 
# ===========================================================

# ---- Global settings ----
knitr::opts_chunk$set(
  echo = TRUE, message = FALSE, warning = FALSE,
  fig.width = 7, fig.height = 5
)

options(repos = c(CRAN = "https://cloud.r-project.org"))
set.seed(123)

# ---- Libraries ----
suppressPackageStartupMessages({
  library(dplyr)
  library(caret)
  library(class)
  library(pROC)
  library(ggplot2)
  library(broom)
  library(knitr)
  library(corrplot)
  library(e1071)
  library(randomForest)
  library(gbm)
  library(pdp)
})

# ---- Project paths  ----
base_dir    <- getwd()
data_dir    <- file.path(base_dir, "data")
plot_dir    <- file.path(base_dir, "figures")
metrics_dir <- file.path(base_dir, "metrics")

dir.create(data_dir, showWarnings = FALSE)
dir.create(plot_dir, showWarnings = FALSE)
dir.create(metrics_dir, showWarnings = FALSE)

# ---- Load data ----
data_path <- file.path(data_dir, "customer_data.csv")
if (!file.exists(data_path)) {
  stop("❌ data/customer_data.csv not found. Add it to the repository.")
}
abc_customer <- read.csv(data_path, stringsAsFactors = FALSE)

# ---- Basic inspection ----
abc_customer$Attrition_Flag <- factor(
  abc_customer$Attrition_Flag,
  levels = c("Existing Customer", "Attrited Customer")
)

# ---- Train / test split (stratified) ----
train_idx <- createDataPartition(
  abc_customer$Attrition_Flag, p = 0.7, list = FALSE
)
train_set <- abc_customer[train_idx, ]
test_set  <- abc_customer[-train_idx, ]

# ---- Data cleaning function ----
clean_data <- function(df, ref = NULL) {
  df <- df %>% mutate(across(where(is.character), as.factor))
  
  num_cols <- names(df)[sapply(df, is.numeric)]
  fac_cols <- names(df)[sapply(df, is.factor)]
  
  for (c in num_cols) {
    med <- if (is.null(ref)) median(df[[c]], na.rm = TRUE)
           else median(ref[[c]], na.rm = TRUE)
    df[[c]][is.na(df[[c]])] <- med
  }
  
  mode_fun <- function(x) names(which.max(table(x)))
  for (c in fac_cols) {
    m <- if (is.null(ref)) mode_fun(df[[c]])
         else mode_fun(ref[[c]])
    df[[c]][is.na(df[[c]])] <- m
    if (!is.null(ref))
      df[[c]] <- factor(df[[c]], levels = levels(ref[[c]]))
  }
  df
}

train_set <- clean_data(train_set)
test_set  <- clean_data(test_set, train_set)

# ---- Binary target ----
train_set$Attrition_Flag_Num <- ifelse(
  train_set$Attrition_Flag == "Attrited Customer", 1, 0
)
test_set$Attrition_Flag_Num <- ifelse(
  test_set$Attrition_Flag == "Attrited Customer", 1, 0
)

# ===========================================================
# Logistic Regression
# ===========================================================
log_model <- glm(
  Attrition_Flag_Num ~ . -CLIENTNUM -Attrition_Flag,
  data = train_set, family = binomial
)
log_prob <- predict(log_model, test_set, type = "response")
log_pred <- factor(
  ifelse(log_prob > 0.5, "Attrited Customer", "Existing Customer"),
  levels = levels(test_set$Attrition_Flag)
)
conf_log <- table(log_pred, test_set$Attrition_Flag)
accuracy_log <- mean(log_pred == test_set$Attrition_Flag)
roc_log <- roc(test_set$Attrition_Flag_Num, log_prob)
auc_log <- auc(roc_log)

# Save ROC plot
png(file.path(plot_dir, "roc_logistic.png"), width = 800, height = 600)
plot(roc_log, main = "ROC Curve: Logistic Regression", col = "blue")
dev.off()

# Save model
saveRDS(log_model, file.path(metrics_dir, "logistic_model.rds"))

# ===========================================================
# Refined Logistic Regression
# ===========================================================
log_step <- step(log_model, trace = FALSE)
ref_prob <- predict(log_step, test_set, type = "response")
ref_pred <- factor(
  ifelse(ref_prob > 0.5, "Attrited Customer", "Existing Customer"),
  levels = levels(test_set$Attrition_Flag)
)
accuracy_ref <- mean(ref_pred == test_set$Attrition_Flag)
roc_ref <- roc(test_set$Attrition_Flag_Num, ref_prob)
auc_ref <- auc(roc_ref)

png(file.path(plot_dir, "roc_refined_logistic.png"), width = 800, height = 600)
plot(roc_ref, main = "ROC Curve: Refined Logistic Regression", col = "green")
dev.off()

saveRDS(log_step, file.path(metrics_dir, "refined_logistic_model.rds"))

# ===========================================================
# KNN
# ===========================================================
X_train <- train_set %>% select(-Attrition_Flag, -Attrition_Flag_Num, -CLIENTNUM)
X_test  <- test_set  %>% select(-Attrition_Flag, -Attrition_Flag_Num, -CLIENTNUM)

X_train <- model.matrix(~.-1, X_train)
X_test  <- model.matrix(~.-1, X_test)
X_test  <- X_test[, colnames(X_train), drop = FALSE]

X_train <- scale(X_train)
X_test  <- scale(X_test,
  center = attr(X_train, "scaled:center"),
  scale  = attr(X_train, "scaled:scale")
)

k <- round(sqrt(nrow(X_train)))
knn_pred <- knn(X_train, X_test, train_set$Attrition_Flag, k = k)
accuracy_knn <- mean(knn_pred == test_set$Attrition_Flag)
auc_knn <- auc(
  roc(test_set$Attrition_Flag_Num,
      ifelse(knn_pred == "Attrited Customer", 1, 0))
)

# ===========================================================
# Naive Bayes
# ===========================================================
nb_model <- naiveBayes(
  Attrition_Flag ~ .,
  data = train_set %>% select(-CLIENTNUM, -Attrition_Flag_Num)
)
nb_prob <- predict(nb_model, test_set, type = "raw")[,2]
nb_pred <- factor(
  ifelse(nb_prob > 0.5, "Attrited Customer", "Existing Customer"),
  levels = levels(test_set$Attrition_Flag)
)
accuracy_nb <- mean(nb_pred == test_set$Attrition_Flag)
auc_nb <- auc(roc(test_set$Attrition_Flag_Num, nb_prob))
saveRDS(nb_model, file.path(metrics_dir, "naive_bayes_model.rds"))

# ===========================================================
# Random Forest
# ===========================================================
rf_model <- randomForest(
  Attrition_Flag ~ .,
  data = train_set %>% select(-CLIENTNUM, -Attrition_Flag_Num),
  ntree = 500, importance = TRUE
)
rf_prob <- predict(rf_model, test_set, type = "prob")[,2]
rf_pred <- predict(rf_model, test_set)
accuracy_rf <- mean(rf_pred == test_set$Attrition_Flag)
auc_rf <- auc(roc(test_set$Attrition_Flag_Num, rf_prob))

# Save variable importance plot
varImpPlot(rf_model, main = "Random Forest Feature Importance")
png(file.path(plot_dir, "rf_importance.png"), width = 800, height = 600)
varImpPlot(rf_model, main = "Random Forest Feature Importance")
dev.off()

saveRDS(rf_model, file.path(metrics_dir, "random_forest_model.rds"))

# ===========================================================
# Gradient Boosting
# ===========================================================
gbm_model <- gbm(
  Attrition_Flag_Num ~ .,
  data = train_set %>% select(-CLIENTNUM, -Attrition_Flag),
  distribution = "bernoulli",
  n.trees = 3000,
  interaction.depth = 3,
  shrinkage = 0.01,
  cv.folds = 5,
  verbose = FALSE
)
best_iter <- gbm.perf(gbm_model, method = "cv", plot.it = FALSE)
gbm_prob <- predict(gbm_model, test_set, best_iter, type = "response")
gbm_pred <- factor(
  ifelse(gbm_prob > 0.5, "Attrited Customer", "Existing Customer"),
  levels = levels(test_set$Attrition_Flag)
)
accuracy_gbm <- mean(gbm_pred == test_set$Attrition_Flag)
auc_gbm <- auc(roc(test_set$Attrition_Flag_Num, gbm_prob))
saveRDS(gbm_model, file.path(metrics_dir, "gbm_model.rds"))

# Save partial dependence plot for top numeric feature
num_features <- names(train_set)[sapply(train_set, is.numeric)]
if(length(num_features) > 1){
  png(file.path(plot_dir, "gbm_pdp_top_feature.png"), width = 800, height = 600)
  pd <- partial(gbm_model, pred.var = num_features[2], n.trees = best_iter)
  plot(pd, main = paste("GBM PDP:", num_features[2]))
  dev.off()
}

# ===========================================================
# Final comparison table & save
# ===========================================================
comparison <- data.frame(
  Model = c("Logistic", "Refined Logistic", "KNN", "Naive Bayes",
            "Random Forest", "Gradient Boosting"),
  Accuracy = round(c(
    accuracy_log, accuracy_ref, accuracy_knn,
    accuracy_nb, accuracy_rf, accuracy_gbm
  ), 4),
  AUC = round(c(
    auc_log, auc_ref, auc_knn,
    auc_nb, auc_rf, auc_gbm
  ), 4)
)
kable(comparison, caption = "Final Model Comparison")

# Save comparison CSV
write.csv(comparison,
  file.path(metrics_dir, "model_comparison.csv"),
  row.names = FALSE
)

# Save predictions
predictions <- data.frame(
  Actual = test_set$Attrition_Flag,
  Logistic = log_pred,
  Ref_Logistic = ref_pred,
  KNN = knn_pred,
  NaiveBayes = nb_pred,
  RandomForest = rf_pred,
  GBM = gbm_pred
)
write.csv(predictions, file.path(metrics_dir, "predictions.csv"), row.names = FALSE)

cat("✅ Analysis complete. All figures, metrics, and models saved to GitHub folders.\n")

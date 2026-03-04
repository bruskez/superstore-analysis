################################################################################
# STEP 3: UNSUPERVISED
################################################################################

# PCA Feature
feature_cols <- c(
  "sales",
  "quantity",
  "profit",
  "discount"
)

X <- data %>%
  select(all_of(feature_cols))

X_scaled <- scale(X)

# =================================================================================
# PCA
# =================================================================================

pca <- prcomp(X_scaled, center = FALSE, scale. = FALSE)
fviz_eig(pca, addlabels = TRUE) + theme_minimal()

# Variables Biplot
fviz_pca_var(pca, col.var = "contrib", gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE) + theme_minimal()

loadings <- pca$rotation
contrib_pct <- 100 * (loadings^2)
print(round(contrib_pct[, 1:4], 1))

# Variables' contribution for PC1 and PC2
fviz_contrib(pca, choice = "var", axes = 1, top = nrow(loadings)) + theme_minimal()
fviz_contrib(pca, choice = "var", axes = 2, top = nrow(loadings)) + theme_minimal()

pca_scores <- as.data.frame(pca$x[, 1:2])
names(pca_scores) <- c("PC1", "PC2")

# =================================================================================
# K-MEANS
# =================================================================================

set.seed(123)

# --- Choosing K (Elbow su K-means) ---
wss <- sapply(1:10, function(k) kmeans(pca_scores, centers = k, nstart = 50)$tot.withinss)

plot(1:10, wss, type = "b",
     xlab = "Number of clusters K", ylab = "Total within-cluster SS",
     main = "Elbow Plot to Choose K")

K <- 4

km <- kmeans(pca_scores, centers = K, nstart = 50)

cluster_cols <- rep(
  c("#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#FFFF33", "#A65628", "#F781BF", "#999999"),
  length.out = K
)

# Clusters visualization
fviz_cluster(
  km, data = pca_scores,
  palette = "clusters_cols",
  geom = "point",
  ellipse.type = "convex",
  ggtheme = theme_minimal()
)

# =================================================================================
# Profiling K-Means boxplots

labels_km   <- km$cluster
prof_km_df <- as.data.frame(X) %>% mutate(cluster = factor(labels_km))

# Cluster means
cluster_means <- prof_km_df %>%
  group_by(cluster) %>%
  summarise(
    mean_sales         = round(mean(sales), 2),
    mean_quantity      = round(mean(quantity), 2),
    mean_profit        = round(mean(profit), 4),
    mean_discount      = round(mean(discount), 4),
    n = n()
  )
cat("\n--- Means for Cluster  (K-means) ---\n")
print(as.data.frame(cluster_means))

# Mean visualization
op_bp <- par(no.readonly = TRUE)
par(mfrow = c(1, 2))
plots <- list(
  list(formula = log1p(sales) ~ cluster, main = "Sales Volume Distribution", ylab = "Log(Sales)"),
  list(formula = quantity ~ cluster,     main = "Quantity by Cluster",       ylab = "Quantity"),
  list(formula = asinh(profit) ~ cluster,main = "Profit by Cluster",         ylab = "Profit (asinh scale)"),
  list(formula = discount ~ cluster,     main = "Discount by Cluster",       ylab = "Discount")
)
for (p in plots) {
  boxplot(p$formula, data = prof_km_df,
          main = p$main, xlab = "Cluster", ylab = p$ylab,
          col = cluster_cols)
}
par(op_bp)

# Categorical variables 
cat_vars <- c("region", "segment", "category", "ship_mode", "sub_category")

# Profiling
for (v in cat_vars) {
  tab <- table(Cluster = labels_km, Livello = data[[v]])
  pct <- round(prop.table(tab, margin = 1) * 100, 1)
  cat("\n--- Distribution %", v, "per Cluster ---\n")
  print(as.data.frame.matrix(pct))
}

# Visualization
plots <- list()

for (v in cat_vars) {
  tab <- table(Cluster = labels_km, Livello = data[[v]])
  pct <- as.data.frame(prop.table(tab, margin = 1) * 100)
  colnames(pct) <- c("Cluster", "Livello", "Percentage")
  pct$Cluster <- factor(pct$Cluster)
  
  p <- ggplot(pct, aes(x = Livello, y = Percentage, fill = Cluster)) +
    geom_bar(stat = "identity", position = "dodge") +
    scale_fill_manual(values = cluster_cols) +
    labs(
      title = paste("Distribution %", v, "by Cluster"),
      x = v, y = "% (row-wise)"
    ) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 30, hjust = 1))
  
  plots[[v]] <- p
}

# 2 rows, 2 plots side by side per row
(plots[[1]] | plots[[2]])
(plots[[3]] | plots[[4]])
par(mfrow = c(1, 1))

# =================================================================================
# HIERARCHICAL CLUSTERING
# =================================================================================
d_x <- dist(pca_scores, method = "euclidean")

# Ward.d2 method Dendrogram
hc_ward <- hclust(d_x, method = "ward.D2")

dend <- as.dendrogram(hc_ward)
dend_col <- color_branches(dend, k = K, col = cluster_cols)
# Visualization
plot(dend_col, leaflab = "none", main = "Hierarchical clustering (Ward.D2)")
rect.hclust(hc_ward, k = K, border = cluster_cols)

# Complete method Dendrogram
hc_comp <- hclust(d_x, method = "complete")

# Average method Dendrogram
hc_avg <- hclust(d_x, method = "average")

# =================================================================================
# K-MEANS vs HIERARCHICAL COMPARISON
# =================================================================================

# Extract cluster labels from the three hierarchical methods for comparison
labels_ward <- cutree(hc_ward, k = K)
labels_comp <- cutree(hc_comp, k = K)
labels_avg  <- cutree(hc_avg, k = K)

# Compute ARI to quantify agreement between K-means and each hierarchical method
cat("ARI Ward:", adjustedRandIndex(labels_km, labels_ward), "\n")
cat("ARI Complete:", adjustedRandIndex(labels_km, labels_comp), "\n")
cat("ARI Average:", adjustedRandIndex(labels_km, labels_avg), "\n")

# Silhouette
methods <- list(
  "K-means" = labels_km, "Ward.D2" = labels_ward,
  "Complete" = labels_comp, "Average" = labels_avg
)
sil_summary <- sapply(methods, function(lab) {
  s <- silhouette(lab, d_x)
  round(mean(s[, 3]), 4)
})
cat("\n--- Silhouette media per metodo ---\n")
print(sil_summary)

table(labels_km)
table(labels_ward)
table(labels_comp)
table(labels_avg)

# =================================================================================
# Profiling Ward.D2 boxplots

prof_wd_df <- as.data.frame(X) %>% mutate(cluster = factor(labels_ward))

# Cluster means
cluster_means <- prof_wd_df %>%
  group_by(cluster) %>%
  summarise(
    mean_sales         = round(mean(sales), 2),
    mean_quantity      = round(mean(quantity), 2),
    mean_profit        = round(mean(profit), 4),
    mean_discount      = round(mean(discount), 4),
    n = n()
  )
cat("\n--- Means for Cluster  (Ward.D2) ---\n")
print(as.data.frame(cluster_means))

# Mean visualization
op_bp <- par(no.readonly = TRUE)
par(mfrow = c(1, 2))
plots <- list(
  list(formula = log1p(sales) ~ cluster, main = "Sales Volume Distribution", ylab = "Log(Sales)"),
  list(formula = quantity ~ cluster,     main = "Quantity by Cluster",       ylab = "Quantity"),
  list(formula = asinh(profit) ~ cluster,main = "Profit by Cluster",         ylab = "Profit (asinh scale)"),
  list(formula = discount ~ cluster,     main = "Discount by Cluster",       ylab = "Discount")
)
for (p in plots) {
  boxplot(p$formula, data = prof_wd_df,
          main = p$main, xlab = "Cluster", ylab = p$ylab,
          col = cluster_cols)
}
par(op_bp)

# Categorical variables 
cat_vars <- c("region", "segment", "category", "ship_mode", "sub_category")

# Profiling
for (v in cat_vars) {
  tab <- table(Cluster = labels_ward, Livello = data[[v]])
  pct <- round(prop.table(tab, margin = 1) * 100, 1)
  cat("\n--- Distribution %", v, "per Cluster ---\n")
  print(as.data.frame.matrix(pct))
}

# Visualization
plots <- list()

for (v in cat_vars) {
  tab <- table(Cluster = labels_ward, Livello = data[[v]])
  pct <- as.data.frame(prop.table(tab, margin = 1) * 100)
  colnames(pct) <- c("Cluster", "Livello", "Percentage")
  pct$Cluster <- factor(pct$Cluster)
  
  p <- ggplot(pct, aes(x = Livello, y = Percentage, fill = Cluster)) +
    geom_bar(stat = "identity", position = "dodge") +
    scale_fill_manual(values = cluster_cols) +
    labs(
      title = paste("Distribution %", v, "by Cluster"),
      x = v, y = "% (row-wise)"
    ) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 30, hjust = 1))
  
  plots[[v]] <- p
}

# 2 rows, 2 plots side by side per row
(plots[[1]] | plots[[2]])
(plots[[3]] | plots[[4]])
par(mfrow = c(1, 1))






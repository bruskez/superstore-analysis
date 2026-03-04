################################################################################
# STEP 1: SETUP & DATA LOADING (RAW DATA)
################################################################################

library(janitor)
library(dplyr)
library(ggplot2)
library(factoextra)
library(cluster)
library(dendextend)
library(mclust)
library(caret)
library(glmnet)
library(ggcorrplot)
library(randomForest)
library(gbm)
library(xgboost)
library(pdp) 
library(sf)
library(tigris)
library(scales)
library(tidyr)
library(patchwork)
library(pROC)

# Path dataset
file_path <- "C:\\file_path\\SampleSuperstore.csv"

data_raw <- read.csv(file_path, stringsAsFactors = FALSE)
str(data_raw)

################################################################################
# STEP 2: DATA CLEANING
################################################################################
data <- data_raw %>%
  clean_names() %>%
  select(-any_of(c("country", "product_name", "postal_code"))) %>%
  mutate(
    ship_mode    = as.factor(ship_mode),
    segment      = as.factor(segment),
    region       = as.factor(region),
    state        = as.factor(state),
    city         = as.factor(city),
    category     = as.factor(category),
    sub_category = as.factor(sub_category),
    loss_flag    = factor(ifelse(profit < 0, 1, 0))
  )

 
################################################################################
# STEP 3: EDA
################################################################################
# Calculate correlation matrix
num_vars <- c("sales", "profit", "discount", "quantity")
data_num <- data %>% select(all_of(num_vars))
corr_mat <- cor(data_num, use = "complete.obs")
ggcorrplot(corr_mat, lab = TRUE, tl.cex = 9, lab_size = 3.3)

# Bar Plot: class imbalance di loss_flag
ggplot(data, aes(x = loss_flag, fill = loss_flag)) +
  geom_bar(show.legend = FALSE) +
  geom_text(stat = "count", aes(label = after_stat(count)), vjust = -0.5, size = 4) +
  scale_fill_manual(values = c("0" = "#4DAF4A", "1" = "#E41A1C")) +
  labs(title = "Target Imbalance (loss_flag)",
       x = "loss_flag (0 = Profit, 1 = Loss)", y = "Count") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14))

# Bar chart: Mean Profit by Discount Band
profit_by_discount <- data %>%
  mutate(discount_bin = cut(discount,
                            breaks = c(-Inf, 0, 0.1, 0.2, 0.3, 0.5, Inf),
                            labels = c("0%", "1-10%", "11-20%", "21-30%", "31-50%", ">50%"))) %>%
  group_by(discount_bin) %>%
  summarise(mean_profit = mean(profit), .groups = "drop")

ggplot(profit_by_discount, aes(x = discount_bin, y = mean_profit, fill = mean_profit > 0)) +
  geom_col(show.legend = FALSE) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey40") +
  geom_text(aes(label = round(mean_profit, 1),
                vjust = ifelse(mean_profit > 0, -0.5, 1.3)), size = 3.5) +
  scale_fill_manual(values = c("TRUE" = "#4DAF4A", "FALSE" = "#E41A1C")) +
  labs(title = "Average Profit by Discount Band",
       x = "Discount Band", y = "Mean Profit") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14))

# ========================================================================================

options(tigris_use_cache = TRUE)

# State Aggregation
by_state <- data %>%
  group_by(state = as.character(state), region = as.character(region)) %>%
  summarise(sales  = sum(sales,  na.rm = TRUE),
            profit = sum(profit, na.rm = TRUE),
            .groups = "drop")

# US Geometry (Alaska e Hawaii excluded)
states_sf <- tigris::states(cb = TRUE, year = 2022, class = "sf") %>%
  filter(STUSPS %in% state.abb, !STUSPS %in% c("AK", "HI")) %>%
  mutate(state = NAME)

states_map <- left_join(states_sf, by_state, by = "state")

# National border
us_border <- st_union(states_sf)

# Map theme
theme_map <- theme_void() +
  theme(legend.position  = "right",
        plot.title        = element_text(face = "bold", hjust = 0.5,
                                         margin = ggplot2::margin(b = 10)),
        plot.margin       = ggplot2::margin(10, 10, 10, 10))


# Regions visualization
ggplot(states_map) +
  geom_sf(aes(fill = region), color = "white", linewidth = 0.2) +
  coord_sf(datum = NA) +
  scale_fill_brewer(palette = "Set2", na.value = "grey85") +
  labs(title = "Regions") +
  theme_map

# Profit per Region
region_profit <- states_map %>%
  group_by(region) %>%
  summarise(profit = mean(profit, na.rm = TRUE), .groups = "drop")

ggplot(region_profit) +
  geom_sf(aes(fill = profit), color = "black", linewidth = 0.15) +
  geom_sf(data = us_border, fill = NA, color = "black", linewidth = 0.6) +
  coord_sf(datum = NA) +
  scale_fill_gradient2(low = "#b2182b", mid = "white", high = "#2166ac",
                       midpoint = 0,
                       labels = dollar_format(prefix = "$"),
                       na.value = "grey90") +
  labs(title = "Profit per Region", fill = "Profit") +
  theme_map

# % Loss transactions by Region
loss_by_region <- data %>%
  group_by(region) %>%
  summarise(pct_loss = round(100 * mean(profit < 0), 1), .groups = "drop")

ggplot(loss_by_region, aes(x = reorder(region, pct_loss), y = pct_loss)) +
  geom_col(fill = "#E41A1C") +
  geom_text(aes(label = paste0(pct_loss, "%")), hjust = -0.2, size = 3.5) +
  coord_flip() +
  labs(title = "% Loss transactions by Region", x = NULL, y = "% transactions with Profit < 0") +
  ylim(0, max(loss_by_region$pct_loss) * 1.15) +
  theme_minimal()


# Profit per Category
catg <- data %>%
  group_by(category) %>%
  summarise(profit = sum(profit, na.rm = TRUE), .groups = "drop")

ggplot(catg, aes(x = reorder(category, profit), y = profit)) +
  geom_col(fill = "steelblue") + coord_flip() +
  labs(title = "Profit per Category", x = NULL, y = "Profit") +
  theme_minimal()

# ========================================================================================
# % Loss transactions by Region
loss_by_region <- data %>%
    group_by(region) %>%
    summarise(pct_loss = round(100 * mean(profit < 0), 1), .groups = "drop")

ggplot(loss_by_region, aes(x = reorder(region, pct_loss), y = pct_loss)) +
    geom_col(fill = "#E41A1C") +
    geom_text(aes(label = paste0(pct_loss, "%")), hjust = -0.2, size = 3.5) +
    coord_flip() +
    labs(title = "% Loss transactions by Region", x = NULL, y = "% transactions with Profit < 0") +
    ylim(0, max(loss_by_region$pct_loss) * 1.15) +
    theme_minimal()

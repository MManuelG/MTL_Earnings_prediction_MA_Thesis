
rm(list=ls())
library(dplyr)
library(ggplot2)
library(pROC)
library(DescTools)
library(xgboost)

# for my analysis I will be using a compustat annual fundamentals dataset and replicating
# Jones et al. (2023), here: https://doi.org/10.1111/1911-3846.12888
# this paper predicts RNOA changes using regression, I will be forecasting the sign
# this R-Script serves as preparation for my masters thesis in Multi-Task Learning
compustat_all <- read.csv("compustat_all.csv")



# drop all obs with NAs in the columns relevant for features or targets!
rows_before <- nrow(compustat_all)
cols_to_check <- c("at", "che", "dlc", "dltt", "mib", "pstk", "ceq", "oiadp", "sale", "ib", "oancf")
compustat_all <- compustat_all[complete.cases(compustat_all[, cols_to_check]), ]
rows_after = nrow(compustat_all)

rows_after / rows_before # => 66% remaining! almost 300k Obs 



#### Filtering as in Jones (2023) ----

# zuerst variablen berechnen
compustat_all$NOA <- NA
compustat_all$OpAss <- NA
compustat_all$OpLia <- NA

# NOA
compustat_all$OpAss <- compustat_all$at - compustat_all$che
compustat_all$OpLia <- compustat_all$at - compustat_all$dlc - compustat_all$dltt - compustat_all$mib - compustat_all$pstk - compustat_all$ceq
compustat_all$NOA <- compustat_all$OpAss - compustat_all$OpLia

compustat_all$PM <- compustat_all$oiadp / compustat_all$sale


# all other vars
compustat_all <- compustat_all %>%
  arrange(cusip, fyear) %>%
  group_by(cusip) %>%
  mutate(
    # Lagged NOA
    NOA_lag = lag(NOA),
    
    # G_NOA
    G_NOA = (NOA - NOA_lag) / (NOA_lag),
    
    # RNOA and change in RNOA
    RNOA = oiadp / NOA_lag,
    RNOA_lag = lag(RNOA),
    CHG_RNOA = RNOA - RNOA_lag,
    
    # Change in PM
    PM_lag = lag(PM),
    CHG_PM = PM - PM_lag,
    
    # ATO and change in ATO
    ATO = sale / NOA_lag,
    ATO_lag = lag(ATO),
    CHG_ATO = ATO - ATO_lag,
    
    # Accruals
    ACC = (ib - oancf) / NOA_lag,
    
    CHG_RNOA_tp1 = lead(CHG_RNOA),
    SALES_GROWTH = (sale - lag(sale))/lag(sale)
    
  ) %>%
  ungroup()

compustat_all <- compustat_all[
  !(compustat_all$sic >= 6000 & compustat_all$sic <= 6999) &      # SIC not in 6000–6999
    compustat_all$NOA >= 0 &                                         # NOA not negative
    abs(compustat_all$SALES_GROWTH) <= 0.5,                          # abs(SALES_GROWTH) <= 0.5
]


# drop all obs with NAs in the columns relevant for features or targets!
rows_before_2 <- nrow(compustat_all)
cols_to_check <- c('CHG_RNOA_tp1', 'ACC', 'CHG_ATO', 'G_NOA', 'CHG_PM', 'CHG_RNOA', 'RNOA')
compustat_all <- compustat_all[complete.cases(compustat_all[, cols_to_check]), ]
rows_after_2 = nrow(compustat_all)


rows_after_2 / rows_before_2 


#### END OF Filtering Jones (2023) ----



#### avg length of individual time series !!!!  ----
cusip_counts <- compustat_all %>%
  group_by(cusip) %>%
  summarise(n_obs = n()) %>%
  arrange(desc(n_obs))
mean(cusip_counts$n_obs) # => 10 years on average 
min(cusip_counts$n_obs);max(cusip_counts$n_obs)



ggplot(cusip_counts, aes(x = n_obs)) +
  geom_histogram(binwidth = 1, color = "black", fill = "steelblue") +
  labs(
    title = "Histogram of Observations per CUSIP",
    x = "Number of Observations",
    y = "Number of CUSIPs"
  ) +
  theme_minimal()

# print how many unique CUSIPS (firms) each "length-bin" has
obs_distribution <- cusip_counts %>%
  count(n_obs, name = "n_cusips") %>%
  arrange(n_obs)
print(obs_distribution, n=50)


# lets say that the model can only be really well trained on firms that have at least 3 observations, then
# our sample would be this big:

nrow(compustat_all) - obs_distribution$n_cusips[1] - obs_distribution$n_cusips[2]
# 122157 
# this is more than enough :)



# Number of unique CUSIPs per fyear 
cusips_per_year <- compustat_all %>%
  group_by(fyear) %>%
  summarise(unique_cusips = n_distinct(cusip)) %>%
  ungroup()

# Plot 
ggplot(cusips_per_year, aes(x = fyear, y = unique_cusips)) +
  geom_line() +
  geom_point() +
  labs(title = "Number of unique CUSIPs per fyear",
       x = "(fyear)",
       y = "# Unique CUSIPs") +
  theme_minimal()
print(cusips_per_year, n = 50)









#### Modeling ----

# bin var
compustat_all$CHG_RNOA_pos_tp1 <- ifelse(compustat_all$CHG_RNOA_tp1 > 0, 1, 0)

# clean --> +/- remove inf and NA and winsorize!
vars <- c("CHG_RNOA_pos_tp1","CHG_RNOA_tp1", "RNOA", "CHG_RNOA", "CHG_PM", "CHG_ATO", "G_NOA", "ACC")

compustat_all <- compustat_all[apply(compustat_all[vars], 1, function(row) all(is.finite(row))), ]


winsorize_manual <- function(x, p = 0.01) {
  q <- quantile(x, probs = c(p, 1 - p), na.rm = TRUE)
  x[x < q[1]] <- q[1]
  x[x > q[2]] <- q[2]
  return(x)
}
compustat_all[vars] <- lapply(compustat_all[vars], winsorize_manual)


# compute model
model <- glm(CHG_RNOA_pos_tp1 ~ RNOA + CHG_RNOA + CHG_PM + CHG_ATO + G_NOA + ACC, 
             data = compustat_all, 
             family = binomial(link = "logit"))
summary(model)



# Predict probabilities
probs <- predict(model, type = "response")

# Compute ROC and AUC
roc_obj <- roc(compustat_all$CHG_RNOA_pos_tp1, probs)

# Print AUC
auc(roc_obj)

plot(roc_obj, col = "blue", main = "ROC Curve")






# OUT OF SAMPLE TEST

# Define predictor and target variables
vars <- c("RNOA", "CHG_RNOA", "CHG_PM", "CHG_ATO", "G_NOA", "ACC")
target <- "CHG_RNOA_pos_tp1"

# Step 1: Split data
train_data <- subset(compustat_all, fyear >= 1993 & fyear <= 2011)
test_data  <- subset(compustat_all, fyear == 2013)

# Step 2: Drop rows with NA in any predictor or target (you could also impute)
train_data <- na.omit(train_data[, c(vars, target)])
test_data  <- na.omit(test_data[, c(vars, target)])

# Step 3: Train logistic model
model <- glm(CHG_RNOA_pos_tp1 ~ RNOA + CHG_RNOA + CHG_PM + CHG_ATO + G_NOA + ACC,
             data = train_data,
             family = binomial)

# Step 4: Predict on 2004
probs <- predict(model, newdata = test_data, type = "response")

# Step 5: Compute AUC
roc_obj <- roc(test_data[[target]], probs)

# Step 6: Print and plot AUC
print(auc(roc_obj))  # numeric value
plot(roc_obj, col = "blue", main = "ROC Curve for 2004 Prediction")


# AUC 0.6519 - consistent (but worse) with results of Jones et al. (2023)!

# if one chooses the same time intervals (train and test) as in Jones et al. (2023):
# 1/7 * (auc1+auc2+auc3+auc4+auc5+auc6+auc7)
# [1] 0.6445429

# jones finds 0.684 for the logistic regression model


# XGBOOST VARIANT


# Step 1: Split data
vars <- c("RNOA", "CHG_RNOA", "CHG_PM", "CHG_ATO", "G_NOA", "ACC")
target <- "CHG_RNOA_pos_tp1"

train_data <- subset(compustat_all, fyear >= 1993 & fyear <= 2011)
test_data  <- subset(compustat_all, fyear == 2013)

# Step 2: Drop rows with NA
train_data <- na.omit(train_data[, c(vars, target)])
test_data  <- na.omit(test_data[, c(vars, target)])

# Step 3: Prepare matrices for XGBoost
dtrain <- xgb.DMatrix(data = as.matrix(train_data[, vars]), label = train_data[[target]])
dtest  <- xgb.DMatrix(data = as.matrix(test_data[, vars]), label = test_data[[target]])

# Step 4: Train XGBoost model (binary:logistic)
xgb_model <- xgboost(data = dtrain,
                     objective = "binary:logistic",
                     eval_metric = "auc",
                     nrounds = 100,
                     verbose = 0)

# Step 5: Predict probabilities
probs <- predict(xgb_model, dtest)

# Step 6: Compute AUC
roc_obj <- roc(test_data[[target]], probs)

# Step 7: Print and plot AUC
print(auc(roc_obj))  # numeric AUC value
plot(roc_obj, col = "blue", main = "ROC Curve for 2022 Prediction (XGBoost)")




# the results are slightly worse than in Jones et al. (2023) but comparable!




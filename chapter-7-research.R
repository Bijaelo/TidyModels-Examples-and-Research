
###
## Chapter 7
###

# Setup
## Load necessary packages

load_p <- function(x, character.only = TRUE, ...){
  if(!require(x, character.only = character.only, ...))
    install.packages(x, ...)
  library(x, character.only = character.only, ...)
}
load_p('tidymodels')
load_p('recipes')
load_p('modeldata') # loaded by tidymodels
load_p('dplyr')
load_p('themis')
load_p('broom')
load_p('parsnip')
load_p('ranger')

## Import dataset
data('ames', package = 'modeldata')
set.seed(123)
ames_split <- initial_split(ames, 0.8)
ames_train <- analysis(ames_split)
ames_test <- assessment(ames_split)
ames_rec <- recipe(Sale_Price ~ ., ames_train) %>%
  step_log(Sale_Price, Gr_Liv_Area, base = 10) %>%
  prep()
ames_train_prep <- juice(ames_rec)
ames_test_prep <- bake(ames_rec, new_data = ames_test)

# 7.1: Create a model
## Models in R are usually fit using a formula interface which creates a model matrix and response vector
## however some interfaces only accept the matrix and response vector such as glmnet, and this is difference can make it frustrating to work with.
## The parsnip package provides a unified interface which is based upon a simple idea

## 1. One should specify the type of model, based on its mathematical structure
## 2. One then specifies the engine, or software, that should be used for the model
## 3. One should specify (where necessary) the outcome, or mode, of the model.

## For example
### Linear model using stats::lm
linear_reg() %>% set_engine('lm')
### linear model using regularized lm in glnet::glmnet
linear_reg() %>% set_engine('glmnet')
### Linear model using regularized bayesian engine in rstan
linear_reg() %>% set_engine('stan')

## The translate function can be used to print a visualization of the call performed.
linear_reg() %>% set_engine('stan') %>% translate()

## We can fit using either fit(...) (for formulas) or fit_xy(x, y, ...) for matrix and vector pairs.
lm_m <- linear_reg() %>%
  set_engine('lm')
lm_f_fit <- lm_m %>%
  fit(Sale_Price ~ Longitude + Latitude, data = ames_train_prep)
lm_xy_fit <- lm_m %>%
  fit_xy(x = ames_train_prep %>% select(Longitude, Latitude),
         y = ames_train_prep %>% pull(Sale_Price))

### Note: The data here is scaled, the datai nthe book is.. not?
lm_f_fit
lm_xy_fit


## Arguments are standardized across packages
## For example, for tree models, mtry, trees and min_n are used for sampled predictors, number of trees and minimum points for the splitting criterium

rand_forest(trees = 1e4, min_n = 5) %>%
  set_engine('ranger') %>%
  set_mode('regression') %>%
  translate()

## As we can see there are multiple arguments added by default
## But sometimes we might want to change these
## For example we might want to change num.threads from 1 to 2
## This can be done in `set_engine`
rand_forest(trees = 1e4, min_n = 5) %>%
  set_engine('ranger', num.threads = 2) %>%
  set_mode('regression') %>%
  translate()


# 7.2: Use the model results

## We can extract the specific parts of the fit using either standard S3 extractors ($, [[]], etc)
lm_f_fit %>% pluck('fit') %>% anova()
lm_f_fit$fit
lm_f_fit %>% pluck('spec')
lm_f_fit[['spec']]


## The broom::tidy function can be used on the parsnip object rather than the fit itself
tidy(lm_f_fit)

# 7.3: Make predictions

## The parsnip package also allows one to make predictions directly from the parsnip object
## There is a few difference however.

## 1: The result is always a tibble
## 2: the result always has predictable column names (starting with a period ".")
## 3: There are always rows equal to the number of predictors.

## "3:" is especially useful. This mean that even if we have rows with missing values, predict will not remove these automatically.

## Note these differ slighlty due to choice of random number for seed.
ames_test_prep %>%
  slice(1:5) %>%
  predict(lm_f_fit, new_data = .)
## Also note that the order is always consistent.

## This can make it very simple to combine the results into existing data

ames_test_prep %>%
  select(Sale_Price) %>%
  bind_cols(predict(lm_f_fit, new_data = ames_test_prep)) %>%
  bind_cols(predict(lm_f_fit, new_data = ames_test_prep, type = 'pred_int')) %>%
  bind_cols(predict(lm_f_fit, new_data = ames_test_prep, type = 'conf_int'))

## These rules all make it simple to change between models and engines.
## We could for example get predictions using random forest:
lm_r_fit <- rand_forest(trees = 1000, min_n = 5) %>%
  set_engine('ranger') %>%
  set_mode('regression') %>%
  fit(Sale_Price ~ ., data = ames_train_prep)
ames_test_prep %>%
  select(Sale_Price) %>%
  bind_cols(predict(lm_r_fit, new_data = ames_test_prep))

### Note that there is no interval for this prediction method.

# 7.4: Parsnip-adjacent packages

###
## Chapter 10
###

# Setup
## Load necessary packages

load_p <- function(x, character.only = TRUE, ...){
  if(!require(x, character.only = character.only, ...))
    install.packages(x, ...)
  library(x, character.only = character.only, ...)
}
load_p('tidymodels')
load_p('dplyr')
load_p('purrr')
load_p('tidyr')
load_p('parsnip')
load_p('workflows')
load_p('yardstick')
load_p('ggplot2')
load_p('patchwork')
load_p('ggthemes')
load_p('themis')
load_p('wesanderson')
load_p('ranger')

## Import dataset
data('ames', package = 'modeldata')
### Note: In previous chapters I used step_log once instead of twice.
### However as the outcome is generally not through available through testing 
### and production this step has been split into 2.
set.seed(123)
ames_split <- initial_split(ames, prob = 0.80, strata = Sale_Price)
ames_train <- training(ames_split)
ames_test  <-  testing(ames_split)

ames_rec <- 
  recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type + 
           Latitude + Longitude, data = ames_train) %>%
  step_log(Gr_Liv_Area, base = 10) %>% 
  step_log(Sale_Price, base = 10, skip = TRUE) %>%
  step_other(Neighborhood, threshold = 0.01) %>% 
  step_dummy(all_nominal()) %>% 
  step_interact( ~ Gr_Liv_Area:starts_with("Bldg_Type_") ) %>% 
  step_ns(Latitude, Longitude, deg_free = 20)

lm_model <- linear_reg() %>%
  set_engine('lm')
lm_wf <- workflow() %>%
  add_model(lm_model) %>%
  add_recipe(ames_rec)
lm_f <- fit(lm_wf, ames_train) 

#- intro
## In the intro they note that the a model cannot be chosen based on the test set
## This is an important note here: In standard litterature one would use the test
## set for finding the model and then use a validation set for evaluating the chosen models performance.
## The reason for this is that, as we normally test Many models against the test dataset
## the estimated precision becomes more and more insecure (the variance increases for estimates based on the dataset).
## To alleviate this we use a third dataset for the final model evaluation, to obtain the highest level of precision.
## In model resampling we usually refit the model to subsets of our dataset and then 
## average model precision across left-out subsets of the data. 
## This means the entire training set can be used for choosing a model, and afterwards
## the model will be refit using the entire training set to be compared with the validation / test set.


# 10.1: The resubstitution approach 
## the model used in this chapter is this chapter will be a random forest model.

rf_model <- rand_forest(trees = 1000) %>% 
  set_engine('ranger') %>%
  set_mode('regression')
rf_wf <- workflow() %>%
  add_formula(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type + Latitude + Longitude) %>% 
  add_model(rf_model)
rf_fit <- fit(rf_wf, data = ames_train %>% mutate(Sale_Price = log10(Sale_Price)))

estimate_perf <- function(model, dat) {
  # Capture the names of the objects used
  cl <- match.call()
  obj_name <- as.character(cl$model)
  data_name <- as.character(cl$dat)
  data_name <- gsub("ames_", "", data_name)
  
  # Estimate these metrics:
  reg_metrics <- metric_set(rmse, rsq)
  model %>% 
    predict(dat) %>% 
    bind_cols(dat %>% select(Sale_Price)) %>% 
    reg_metrics(Sale_Price, .pred) %>% 
    select(-.estimator) %>% 
    mutate(object = obj_name, data = data_name)
}
# Note that Sale_Price here is not log transformed.
ames_train_old <- ames_train
ames_train <- ames_train %>% mutate(Sale_Price = log10(Sale_Price))
estimate_perf(rf_fit, ames_train)
estimate_perf(lm_f, ames_train)


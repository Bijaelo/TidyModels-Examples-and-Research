###
## Chapter 11
###

# Setup
## Load necessary packages

load_p <- function(x, character.only = TRUE, ...){
  if(!require(x, character.only = character.only, ...)){
    install.packages(x, ...)
    Recall(x = x, character.only = character.only, ...)
  }
}
load_p('tidymodels')
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
load_p('dplyr')

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

rf_model <- rand_forest(trees = 1000) %>% 
  set_engine('ranger') %>%
  set_mode('regression')
rf_wf <- workflow() %>%
  add_formula(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type + Latitude + Longitude) %>% 
  add_model(rf_model)

ames_folds <- vfold_cv(ames_train, 10)

rf_res <- rf_wf %>% fit_resamples(ames_folds)


# 11.1: Resampled performance statistics
## Lets consider the situation where we want to compare two
## or more models together.
lm_wsr <- fit_resamples(lm_wf, resamples = ames_folds, control = control_resamples(save_pred = TRUE))



ames_ns_rec <- 
  recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type + 
           Latitude + Longitude, data = ames_train) %>%
  step_log(Gr_Liv_Area, base = 10) %>% 
  step_log(Sale_Price, base = 10, skip = TRUE) %>%
  step_other(Neighborhood, threshold = 0.01) %>% 
  step_dummy(all_nominal()) %>% 
  step_interact( ~ Gr_Liv_Area:starts_with("Bldg_Type_") ) 
lm_ns_res <- 
  lm_wf %>% 
  remove_recipe() %>%
  add_recipe(ames_ns_rec) %>%
  fit_resamples(resamples = ames_folds, control = control_resamples(save_pred = TRUE))


collect_metrics(lm_ns_res)
collect_metrics(lm_wsr)



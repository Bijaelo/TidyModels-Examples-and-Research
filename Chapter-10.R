###
## Chapter 10
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
    bind_cols(dat %>% dplyr::select(Sale_Price)) %>% 
    reg_metrics(Sale_Price, .pred) %>% 
    select(-.estimator) %>% 
    mutate(object = obj_name, data = data_name)
}
# Note that Sale_Price here is not log transformed.
ames_train_old <- ames_train
ames_train <- ames_train %>% mutate(Sale_Price = log10(Sale_Price))
estimate_perf(rf_fit, ames_train)
estimate_perf(lm_f, ames_train)

ames_test_old <- ames_test
ames_test <- mutate(ames_test, Sale_Price = log10(Sale_Price))
estimate_perf(rf_fit, ames_test)


# 10.2: Resampling methods
set.seed(55)
ames_folds <- vfold_cv(ames_train, v = 10)
ames_folds
library(pryr)
object_size(ames_folds)
## interesting.. There might be some problems with memory here.
## The data is stored in .$splits[[i]]$data and splits is a list not environment.
## So it would be interesting to see if data takes up v times the memory after usage.

object_size(ames_folds$splits[[1]]$data)
format(object.size(ames_folds$splits[[1]]$data), unit = 'KiB')
## Oh they note this right after the split, and state R is smart enough.
## I guess this is due to the promis that the data should be available, until it is altered.

#- Repeated cross validation

vfold_cv(ames_train, v = 10, repeats = 5)

#- LOOCV
loo_cv(ames_train)

#- Monte carlo CV
## Interesting, didn't know of this one, but basically the samples are taken at random.
## So the assessment might contain multiple of the same data point
mc_cv(ames_train, prop = 9/10, times = 20)

#- Validation set
## exactly the same as "initial split" but with a different class
validation_split(ames_train)

#- Bootstrapping
## make splits with replacements. Any points not chosen in analysis will be used for assessment
## This maens assessment can have 0 to k points
bootstraps(ames_train, times = 5)

#- Rolling forecasting origin resampling
## Simply roll across the time series using blocks of data to predict equally or unequally sized blocks in the future.
time_slices <- 
  tibble(x = 1:365) %>% 
  rolling_origin(initial = 6 * 30, assess = 30, skip = 29, cumulative = FALSE)

data_range <- function(x) {
  summarize(x, first = min(x), last = max(x))
}

map_dfr(time_slices$splits, ~ assessment(.x) %>% data_range())

# 10.3: Estimating performance
## We can use rsample object to simply fit our models using fit_resample

rf_wf %>% fit_resamples(ames_folds)
## We can also use this interface with a model_spec using either a formula interface or 
## by using a recipe interface

spec <- rand_forest(trees = 1000) %>% 
  set_engine('ranger') %>%
  set_mode('regression')
spec %>% fit_resamples(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type + Latitude + Longitude, 
                      ames_folds)
spec %>% fit_resamples(ames_rec, ames_folds)

## The fit_resamples method has a "metrics" argument which can be used to specify which metrics to use
## control can further be used to specify further parameters. 
## The parameters are listed in ?control_resamples
keep_pred <- control_resamples(save_pred = TRUE)

set.seed(130)
rf_res <- 
  rf_wf %>% 
  fit_resamples(resamples = ames_folds, 
                metrics = metric_set(rsq, rmse), 
                control = keep_pred)
rf_res

## Metrics can be collected using collect_metrics
collect_metrics(rf_res)

## Predictions can be collected using collect_predictions
assess_res <- collect_predictions(rf_res)
assess_res
## For cross validation where observations are sampled with replacement 
## summarize can be set to TRUE in collect_predictions. (Does nothing here)
collect_predictions(rf_res, summarize = TRUE)

assess_res %>% 
  ggplot(aes(x = Sale_Price, y = .pred)) + 
  geom_point(alpha = .15) +
  geom_abline(col = "red") + 
  coord_obs_pred() + 
  ylab("Predicted") + 
  theme_pander()

## agai we see one observation that is over predicted
over_predicted <- assess_res %>%
  mutate(.resid = Sale_Price - .pred) %>%
  arrange(desc(abs(.resid))) %>%
  slice(1)
over_predicted

ames_train %>% 
  slice(over_predicted$.row) %>% 
  select(Gr_Liv_Area, Neighborhood, Year_Built, Bedroom_AbvGr, Full_Bath)

set.seed(12)
val_set <- validation_split(ames_train, prop = 3/4)

val_res <- rf_wf %>% fit_resamples(resamples = val_set)
val_res
collect_metrics(val_res)

# 10.4: Parallel Processing
## The tidymodels universe can be parallerized using various "do*" packages
## doMC can be used for forking
## doParallel and doFuture can be used alternatively.

load_p('future')
load_p('doFuture')
load_p('foreach')
load_p('doMC')
load_p('doParallel')

nc <- parallel::detectCores()
registerDoMC(cores = 2)
rf_wf %>% 
  fit_resamples(resamples = ames_folds, 
                metrics = metric_set(rsq, rmse), 
                control = keep_pred)
registerDoSEQ()

cl <- makeCluster(nc - 1)
registerDoParallel(cl)
rf_wf %>% 
  fit_resamples(resamples = ames_folds, 
                metrics = metric_set(rsq, rmse), 
                control = keep_pred)
stopCluster(cl)
registerDoSEQ()

registerDoFuture()
rf_wf %>% 
  fit_resamples(resamples = ames_folds, 
                metrics = metric_set(rsq, rmse), 
                control = keep_pred)
registerDoSEQ()

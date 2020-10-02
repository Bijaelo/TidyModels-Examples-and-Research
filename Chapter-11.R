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
load_p('corrr')
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

## We could do this by comparing the metrics 
## between the fitted models
## But as we should already know, these metrics have a tendency to be somewhat correlated
collect_metrics(lm_ns_res)
collect_metrics(lm_wsr)

## If we collect the individual metrics from each resample
## we can illustrate this correlation
ns_rsq <- collect_metrics(lm_ns_res, summarize = FALSE) %>%
  filter(.metric == 'rsq') %>%
  select(id, 'no splines' = .estimate)
s_rsq <- collect_metrics(lm_wsr, summarize = FALSE) %>%
  filter(.metric == 'rsq') %>%
  select(id, 'with splines' = .estimate)
rf_rsq <- collect_metrics(rf_res, summarize = FALSE) %>%
  filter(.metric == 'rsq') %>%
  select(id, 'random forest' = .estimate)

rsq_estimates <- 
  inner_join(ns_rsq, s_rsq, by = 'id') %>%
  inner_join(rf_rsq, by = 'id')

## this shows clearly that there is quite a bit of correlation
## We could test whether this is significant as well
corrr::correlate(rsq_estimates %>% select(-id))
rsq_estimates %>%
  with(cor.test(`no splines`, `random forest`)) %>%
  tidy() %>%
  select(estimate, starts_with('conf'))
## And clear this is significant

## We could also visualize the effects
## And it is quite clear that several lines
## tend to follow the same trend across different folds
rsq_estimates %>% 
  pivot_longer(cols = c(-id), 
               names_to = "model", 
               values_to = "rsq") %>% 
  mutate(model = reorder(model, rsq)) %>% 
  ggplot(aes(x = model, y = rsq, group = id, col = id)) + 
  geom_line(alpha = .5, lwd = 1.25) + 
  labs(x = NULL, y = expression(paste(R^2, "statistics"))) + 
  theme_pander() +
  theme(legend.position = "none") 

## Now this is something I haven't worked with in this scenario, but obviously if we were to 
## to perform a statistical test to evaluate which model were performing better (using a test on the difference on the estimate)
## this would cause the result to be biased as they are not independent

# 11.2: Simple hypothesis test

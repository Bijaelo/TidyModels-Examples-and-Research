###
## Chapter 8
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
load_p('workflow')
load_p('yardstick')
load_p('ggplot2')
load_p('patchwork')
load_p('ggthemes')
load_p('themis')

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
# 9.1: Performance Metrics and Inference
## This chapter primarily focused on how we can evaluate the accuracy of our models, 
## and not on the inferential aspects, where we might be more concerned by the
## theoretical validity of the model and the interpretation of the results.


# 9.2: Regression Metrics
## Predictions using predict on workflow object
ames_test_res <- predict(lm_f, new_data = ames_test)

ames_test_res <- bind_cols(ames_test %>% mutate(Sale_Price = log10(Sale_Price)) %>% select(Sale_Price),
                           ames_test_res)
ggplot(ames_test_res, aes(x = Sale_Price, y = .pred)) + 
  geom_point(alpha = .5) +
  geom_abline(lty = 2) +
  labs(y = '(log10) Predicted Sale Price', x = '(log10) Sale Price') + 
  coord_obs_pred() + 
  theme_pander()

## We can calculate various metrics using the implementations from yardstick
rmse(ames_test_res, truth = Sale_Price, estimate = .pred)

## If we want multiple metrics we can specify a `metric_set` for these to be calculated simultaniously
ames_metrics <- metric_set(rmse, rsq, mae)
ames_metrics(ames_test_res, truth = Sale_Price, estimate = .pred)
## A (semi) automatic version of this is contained in the function "metrics" 
## which has some standard choices.
metrics(ames_test_res, truth = Sale_Price, estimate = .pred)

# 9.3: Binary classification metrics
## For binary classification the two_class_example data in modeldata is used as example data.
data("two_class_example", package = 'modeldata')

## The metrics function still works but only provides a few different estimators.
metrics(two_class_example, truth = truth, estimate = predicted)
## We can however also get AUC if we specify class probabilities
metrics(two_class_example, truth = truth, estimate = predicted, Class1)

## Other metrics are available for example we can get log loss, matthews correlation coefficient, 
## F1 score and the confusion matrix
mn_log_loss(two_class_example, truth = truth, Class1)
accuracy(two_class_example, truth = truth, estimate = predicted)
mcc(two_class_example, truth = truth, estimate = predicted)
conf_mat(two_class_example, truth = truth, estimate = predicted)

## Lets try to combine them all into one big jumble
binary_metrics <- metric_set(roc_auc, accuracy, kap, mn_log_loss, mcc, conf_mat)
## Interesting. So these do not work. 
## So we'd have to combine these as.
ev_1 <- bind_rows(
  metrics(two_class_example, truth = truth, estimate = predicted, Class1),
  mcc(two_class_example, truth = truth, estimate = predicted),
  f_meas(two_class_example, truth = truth, estimate = predicted))
ev_1
## and if we wanted the confusion matrix we should likely give it this as sensitivity etc.

## By default the event of interest is assumed to be the first level.
## However this can be changed by specifyin this n the `event_level` argument
## Note in the below that the f1 score is changed
ev_2 <- bind_rows(
  metrics(two_class_example, truth = truth, estimate = predicted, Class1, options = list(event_level = "second")),
  mcc(two_class_example, truth = truth, estimate = predicted, event_level = "second"),
  f_meas(two_class_example, truth = truth, estimate = predicted, event_level = 'second')) 
ev_2 %>% inner_join(ev_1, by = c(".metric", ".estimator"))

## Another useful measure is the ROC curve. This ca be calculated simply by using 
## roc_curve
two_class_curve <- roc_curve(two_class_example, truth, Class1)
two_class_curve

p1 <- autoplot(two_class_curve) + 
  theme_pander()
p2 <- gain_curve(two_class_example, truth, Class1) %>% 
  autoplot()
p3 <- lift_curve(two_class_example, truth, Class1) %>% 
  autoplot()
p4 <- pr_curve(two_class_example, truth, Class1) %>% 
  autoplot()
(p1 + p2) / (p3 + p4) + plot_annotation(title = 'Different measures in yardstick (binary)') 





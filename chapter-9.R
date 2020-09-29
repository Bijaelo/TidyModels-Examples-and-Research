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

# 9.4: Multi-class classification metrics
## For multi-class problems we can continue to use the same methods as for 
## binary classification
data(hpc_cv, package = 'modeldata')
tibble(hpc_cv)
metrics(hpc_cv, obs, pred)

hpc_ms <- metric_set(accuracy, kap, mcc)
hpc_ms(hpc_cv, obs, estimate = pred)

## Some metrics in binary search has approximations for the multi-class case.
## Sensitivity for example has macro-weighted, macro- and micro-averaging
## Macro averaging computes a set of one-versus-all metrics. 
## Eg. For each class, assume this is the "positive". What is the true positive rate given this assumption?

## macro weighted averaging is similar, but weights the estimate by the observation count
## Micro averaging combines the result for each class into a single metric.


#- Macro F1 and Macro F1, a note. (Juri Opitz & Sebastian Burst). Article reference.
## As I've never heard about multiclass F1 and as the theory for the above is not completely clear
## I've decided to go through the references and try to understand the theorems and proofs
## I will make some very inconsistent notes on this and try to make my own calculations for this purpose, before continuing to read.
## Here i use a lax and inconsistent latex notation

## Introduction:
##- Preliminaries (subsection)

### In the article they state two metrics normally are used. They name them
### Averaged F1 and F1 of averages

### For any classifier f : D -> C = {1, ..., n} and finite set S contained within D \times X
### let m^{f,S} in N_0^{n \times n} be a confusion matrix, where 
### m_{i,j}^{n, S} = |{s in S | f(s_1) = i \wedge s_2 = j}| 
### (where the element of a confusion matrix is given by the value of the classifier for subset s_1 given subset s_2 is j.)
### Subscript omitted from here on
### For any such matrix let P_i, R_i and F1_i denote precision, recall and F1-score with respect to class i:
### P_i = m_ii / {\sum_{x = 1}^{n} m_{i,x}}
### R_i = m_ii / {\sum_{x = 1}^{n} m_{x,i}} 
### F1_i = H(P_i, R_i) = 2 P_i R_i / {P_i + R_i}
### with P_i, R_i, F1_i = 0 when the denominator is zero. 
### H is the harmonic mean, precision and recall are also known as positive predictive value and sensitivity

##- (Averaged F1: arithmetic mean over harmonic means)
### F1 scores are computed for each class and then averaged via arithmetic mean
### f1 = 1 / {N} \sum_{x} F1_x = 1/{n} \sum_{x} 2 P_x R_x / {P_x + R_x}

##- (F1 of averages: harmonic mean over arithmetic means)
### The harmonic mean is computed over the arithmetic means of precision and recall

### bF1 = H(\bar P, \bar R) = 2 \bar P \bar R / {\bar P + \bar R} 
###     = 2 * (1 / n \sum_{x} P_{x}) (1 / n \sum_x R_x) / {1/n \sum_{x} P_{x} + 1/n \sum_{x} R_{x}}


#### calculating the two from hpc_cv
#### Lets see if I can get this right

##### We'll start by calculating individual recall, sensitivity and F1 scores

summaries <- hpc_cv %>%
  group_by(obs) %>%
  count(pred) %>%
  summarize(pred = pred,
            n = n,
            recall = n / sum(n)) %>%
  group_by(pred) %>%
  summarize(obs = obs,
            n = n, 
            recall = recall,
            sensitivity = n / sum(n)) %>%
  mutate(F1 = 2 * recall * sensitivity / (recall + sensitivity)) %>%
  filter(pred == obs) %>%
  ungroup()


##### Next we'll just follow the formulas
summaries %>% 
  summarize(f1 = mean(F1),
            bF1 = 2 * mean(recall) * mean(sensitivity) / (mean(recall) + mean(sensitivity)))

##### hmmm..... This did not match the result by tidyverse. 
##### back to the drawing board. Probably my grouping is not good enough.

##### Lets try just getting regular precision and sensitivity right first

summary(hpc_cv)
hpc_small <- hpc_cv %>% filter(obs %in% c('VF', 'F'), pred %in% c('VF', 'F'))

hpc_small %>% 
  group_by(obs, pred) %>% 
  count() %>%
  ungroup() %>%
  group_by(obs) %>%
  mutate(recall = n / sum(n)) %>%
  ungroup() %>%
  filter(pred == obs)

recall(hpc_small %>% mutate(obs = droplevels(obs), pred = droplevels(pred)), 
       truth = obs, estimate = pred, 
       event_level = 'second')

###### Well that one is good enough. Now lets get precision
hpc_small %>% 
  group_by(obs, pred) %>% 
  count() %>%
  ungroup() %>%
  group_by(pred) %>%
  mutate(sens = n / sum(n)) %>%
  ungroup() %>%
  filter(pred == obs)

precision(hpc_small %>% mutate(obs = droplevels(obs), pred = droplevels(pred)), 
       truth = obs, estimate = pred, 
       event_level = 'second')

##### Alright so that is correct... wtf did i do wrong above.
##### Lets get f1 score correct

mij <- hpc_small %>%
  group_by(obs, pred) %>%
  count(obs, pred) %>%
  ungroup()

P <- mij %>%
  group_by(obs) %>%
  mutate(rec = n / sum(n)) %>%
  ungroup() %>%
  filter(pred == obs)
R <- mij %>%
  group_by(pred) %>%
  mutate(sens = n / sum(n)) %>%
  ungroup() %>%
  filter(pred == obs)
F <- full_join(P, R, c('obs', 'pred')) %>%
  mutate(F1 = 2 * rec * sens / (rec + sens) )
F

f_meas(hpc_small %>% mutate(obs = droplevels(obs), pred = droplevels(pred)), 
          truth = obs, estimate = pred, 
          event_level = 'first')

full_join(P, R, c('obs', 'pred')) %>% 
  mutate(F1 = 2 / ( rec^(-1) + sens^(-1)))

##### Alright everything here seems up to snuff. Now lets do it for all groups.
mij <- hpc_cv %>% 
  count(pred, obs) 
P <- mij %>%
  group_by(obs) %>%
  mutate(rec = n / sum(n)) %>%
  ungroup() %>%
  filter(pred == obs)
R <- mij %>%
  group_by(pred) %>%
  mutate(sens = n / sum(n)) %>%
  ungroup() %>%
  filter(pred == obs)
F <- full_join(P, R, c('obs', 'pred', 'n'))
##### So we know these are correct
##### The next part is to take the arithmetic mean over harmonic means and the harmonic mean over arithmetic means
F %>% 
  mutate(F1 = 2 / (1 / rec + 1 / sens)) %>% 
  mutate(f1 = mean(F1)) %>% 
  mutate(bF1 = 2 / ( 1 / mean(rec) + 1 / mean(sens)))

f_meas(hpc_cv, truth = obs, estimate = pred, estimator = 'macro')
f_meas(hpc_cv, truth = obs, estimate = pred, estimator = 'macro_weighted')
f_meas(hpc_cv, truth = obs, estimate = pred, estimator = 'micro')

##### It seems macro_weighted is wrong on my end. I am not entirely sure why that is.
##### It should be simple to replicate... in the function f_meas (if one searches down the call stack) 
##### it just takes a weighted average... But I can't seem to get this one right..
F %>% 
  mutate(F1 = 2 / (1 / rec + 1 / sens)) %>% 
  mutate(f1 = mean(F1)) %>% 
  mutate(bF1 = 2 / ( 1 / mean(rec) + 1 / mean(sens))) %>% 
  group_by(pred) %>% 
  mutate(w = n / sum(n)) %>% 
  ungroup() %>% 
  mutate(bf1 = weighted.mean(F1, w))



sensitivity(hpc_cv, obs, pred, estimator = "macro")
sensitivity(hpc_cv, obs, pred, estimator = "macro_weighted")
sensitivity(hpc_cv, obs, pred, estimator = "micro")


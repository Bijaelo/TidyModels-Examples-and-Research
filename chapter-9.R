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

##### But I can't seem to get the weighing correct. They have a complete example in the 9.4 chapter
class_totals <- 
  count(hpc_cv, obs, name = "totals") %>% 
  mutate(class_wts = totals / sum(totals))
class_totals
cell_counts <- 
  hpc_cv %>% 
  group_by(obs, pred) %>% 
  count() %>% 
  ungroup()
one_versus_all <- 
  cell_counts %>% 
  filter(obs == pred) %>% 
  full_join(class_totals, by = "obs") %>% 
  mutate(sens = n / totals)
one_versus_all

##### But this does not seem to be directly related to the estimate in the referenced article
##### Likely the specific metric comes from one of the references in the f_meas documentation
##### I should go through those at some point as well... Maybe I will.


##### Regardless, the authors of this current article suggest using macro-averaged and not their alternative (harmonic average over arithmetic means). 
##### Likely the reason why my result doesnt match macro-weighed-averaged is that specific suggestion, meaning they are not identical.



# (A unified View of Multi-Label Performance Measures) [article reference]
#- Abstract
## They introduce what they call "label-wise" and "instance-wise" margin classifications measures.
## They come up with a new margin measure "LIMO" they propose using.

#- Introduction
## They claim that many measures have not been studies sufficiently for their consequences
## in the paper they show that to optimize some of these measures (eg. gain the best hamming-loss) 
## certain other measures will simultaneously be optimized while other measure will be gain optimum together as well
## In addition they state that they come with a method for optimizing both "sides" (margins) simultaniously using LIMO.

#- Preliminaries
##- notation
### Assume that x_i \in \real^{d\times 1} is a real value instance vector
### y_i \in {0, 1} ^{l\times 1} is a label vector for x_i
### m denotes the number of training samples
### y_ij (i\in {1, ..., m} j\in{1, ..., l}) means the j'th label of the i'thinstance 
### y_ij = 1 or 0 <=> y_ij is relevant or irrelevant
### X in \real^{m\times d} is an instance matrix
### Y\in {0, 1}^{m\times l} is a label matrix
### H : \real^d -> {0, 1}^l is the multi label classifier, each classifier can be thought of as a model
### Eg. H = {h_1, ... h_l} and h_j(x_i) denotes the prediction of {y_i}_j = y_ij (the class label or probability for label j for y_i ) 
### F : \real^d -> \real^l  is a multi label predictor 
### F(X) can be regarded as the confidence of relevance (whatever that means?)
### F = {f_1, ..., f_l} -> f_j(x_i) is the predicted value of y_ij
### as any other theory, H can usually be induced via some threshold of F
### h_j(x_i) = [[f_j(x_i) > t(x_i)]] where t is some treshold function. 
### (Eg. in binary the threshold is usually whether f_j(x_i) > 0.5, so t(x_i) = 0.5)
### Y_i denotes the i'throw vector of Y, and Y_.j is the j'th column vector
### Y_i.^+ (or Y_i.^-) denotes the index set of relevant (or irrelevan) labels of Y_i
### Eg. Y_i.^+ = {j| y_ij = 1} and Y_i.^- = {j| y_ij = 0}
### |Y_ i.^+|  is used to notate the number of relevant points in x_i


##- Multi-label performance measures
### 11 measures are considered. 
### 3 of them are F-measure extensions
### 3 of them are extensions of AUC
### 5 of them are various measures of error and loss.
### The authors note that there is some ill-defined measures 
### for example coverage(F) = one-error(F) = 0 if F_ij = 1 \forall ij
### Also some of the AUC measures will similarly be 1 in this case.

#### Lets try to calculate all of these measures. 

####- Hamming loss (Fraction of misclassified labels)
##### ah.. that is nice and simple at least
(hpc_cv %>% 
  filter(obs != pred) %>% nrow) /
  nrow(hpc_cv)

####- Ranking loss
##### Darn.. ehm..
##### rloss(F) = 1 / M \sum_{i = 1}^m |S_rank^i| / {|Y_i.^+| * |Y_i.^-|}
##### S_rank^i = {(u,v)| f_u(x_i) \leq f_v(x_i), (u,v) \in Y_i.^0 \times Y_i.^-}
##### ... well.. lets do this. Note that this is meant to be done on predictions
##### In this case our predicted probability is in the VF, F, M and L columns
##### Split Y_i.^+ and Y_i.^- into difference calculations. 
##### Do the same for S_rank^i

##### |Y_i.^+| and |Y_i.^-| is just the number of observations that are in class "i" and those not in class "i"
##### S_rank^i is just a vector of indexes where indicating the positions in the confusion matrix, 
##### where an observation of class "i" is given a lower probability than a class not of rank "i"
##### So this should not be too bad. 
rank_count <- hpc_cv %>% 
  select(obs, VF, F, M, L) %>% 
  mutate(id = seq(n())) %>%
  pivot_longer(c('VF', 'F', 'M', 'L')) %>%
  # Find the number value of match
  group_by(id) %>%
  mutate(truth = max(case_when(obs == name ~ value, 
                           TRUE ~ 0))) %>%
  ungroup() %>%
  group_by(obs) %>%
  summarize(count = sum(truth < value))
 
product_count <- hpc_cv %>% 
  group_by(obs) %>%
  summarize(n = n()) %>%
  mutate(total = nrow(hpc_cv))

left_join(rank_count, product_count, by = 'obs') %>% 
  mutate(rankL_i = count / (n * (total - n))) %>%
  summarize(rankL = mean(rankL_i))
##### this does seem.. oddly low.. But it seems to follow the definition.. 
##### I would take this with a grain of salt, as it it literally saying the model is ranking almost perfectly.


####- One-error
##### Similar to hamming loss, but based on probabilistic measure rather than predicted value

hpc_cv %>% 
  select(obs, VF, F, M, L) %>% 
  mutate(id = seq(n())) %>%
  pivot_longer(c('VF', 'F', 'M', 'L')) %>%
  # Find the number value of match
  group_by(id) %>%
  mutate(truth = max(case_when(obs == name ~ value, 
                               TRUE ~ 0))) %>%
  ungroup() %>%
  filter(obs != name) %>%
  mutate(get = truth < value) %>%
  group_by(obs, id) %>%
  summarize(any_wrong = any(get)) %>%
  ungroup() %>%
  summarize(one_error = mean(any_wrong))
##### interesting that this is almost just 100 times greater than the rank loss estimated.


####- Coverage
##### I dont know... ehm
##### It really seems like it is "just" how many labels are never guessed correctly divded by the number of labels
##### but that would be ignoring the rank... I am not completely familiar with this the ranking notation in this.

#### Maybe we'll just skip calculating each measure, 
#### and come back when I've got more time for it.
#### I've gotta remember more about my linear algebra for this stuff it seems.









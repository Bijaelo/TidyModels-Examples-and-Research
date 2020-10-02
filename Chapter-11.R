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
load_p('tidyposterior')
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
## A simple method for testing whether there is a difference
## in the estimated statistics would be to fit a regression on the estimated 
## measure with the model and fold as dummy variables.
## For example if we wanted to compare the model with and without splines
c_lm <- rsq_estimates %>%
  mutate(difference = `with splines` - `no splines`) %>%
  lm(formula = difference ~ 1, data = .) %>%
  tidy(conf.int = TRUE) %>%
  select(estimate, p.value, starts_with('conf'))
c_lm
## From this we can see that there is seemingly a difference between the two estimates (they may be significant)

## Ofc. we could also get the same result with a paired t.test
## as we know from mixed effect models
with(rsq_estimates, t.test(`with splines`, `no splines`, paired = TRUE)) %>%
       tidy() %>%
       select(estimate, p.value, starts_with('conf'))
  
# 11.3: Bayesian Methods 
## This is (almost) all new to me, I am terribly behind on bayesian statistics. That is a shame.
## In bayesian methods we have to set up prior information, 
## prior information is the information that we assume to be somewhat related to the actual data that we are analysing.
## For example in the model we just fit, a prior information could be the distribution of the various parameters
## In the book they give an example where
## e_ij ~ N(0,\sigma)
## b_j ~ N(0, 10)
## \sigma ~ exp(1)
## which we could consider a somewhat "uninformative" prior
## as the range of b and sigma is quite wide.

## using this prior information we can fit a model, and obtain the posterior distribution
## of the parameters. These parameters are combinations of the prior and maximum likelihood estimates
## To gain the extra benefit from fitting stand models, they suggest using a random intercept model (as an example)
## As they rightly note, this limits the model to note have any variability in intercept, 
## with is not entirely the case as we see in the visual, but 
## 1) we only have 10 observations so additional parameters might be hard to evaluate uniquely
## 2) For the 2 levels of splines and no splines we see only a limited amount of variability in slope.

ames_tm <- bind_cols(ames_folds, 
                     rsq_estimates %>%
                       arrange(id) %>%
                       select(-id))
ames_tm
## We can use the tidyposterior package for this purpose.
## This package is an interface to the rstanarm package, 
## and this package sets a vector of default priors. 
## The `perf_mod` function (from tidyposterior) can be used to fit an appropriate bayesian model
## (note this is all from the book, i have no prior knowledge of rstanarm)

library(rstanarm)
options(mc.cores = parallel::detectCores() - 1)
rsq_anova <- perf_mod(
  ames_tm, 
  prior_intercept = student_t(df = 1),
  chains = parallel::detectCores() * 2,
  iter = 5000, 
  seed = 2
)
## Tidy up the result (seed???)
model_post <- 
  tidy(rsq_anova, seed = 35) %>%
  as_tibble() 

glimpse(model_post)

## Visualize the posterior distribution of Rsquared.
library(forcats)
mutate(model_post, model = fct_inorder(model)) %>%
  ggplot(aes(x = posterior)) +
  geom_histogram(bins = 50, col = 'white',
                 fill = 'blue', alpha = 0.4) + 
  facet_wrap(~ model, ncol = 1) + 
  labs( x = expression(paste("Posterior for mean ", R^2)))
  
## Now that we have the posterior distribution
## we can compare the means by sampling from the distributions (mcmc) and comparing the difference for the samples
rsq_diff <- contrast_models(rsq_anova, 
                            list_1 = "with splines",
                            list_2 = "no splines", 
                            seed = 36)
as_tibble(rsq_diff) %>% 
  ggplot(aes(x = difference)) + 
  geom_vline(xintercept = 0, lty = 2) + 
  geom_histogram(bins = 50, col = "white", fill = "red", alpha = 0.4) + 
  labs(x = expression(paste("Posterior for mean difference in ", R^2, 
                            " (splines - no splines)")))
## We can get a formal test by using summary. Note that "prob" indicates the probability mass, not alpha.
## Eg. our significance is (1 - prob) / 2
summary(rsq_diff, prob = 0.90) %>% 
  select(-starts_with("pract"))

## If we have some practical level of significance that we want to test around 
## this can be specified in summary using the "size" parameter
## This is a rope interval.. Definitely need to read the reference on this
## I have absolutely no idea what that means.
summary(rsq_diff, size = 0.02) %>%
  select(starts_with('pract'))


# The Bayesian New Statistics: Hypothesis testing, estimation meta-analysis and power analysis from a Bayesian Perspective
# (article about ROPE intervals)


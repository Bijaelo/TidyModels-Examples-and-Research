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
load_p('survival')

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
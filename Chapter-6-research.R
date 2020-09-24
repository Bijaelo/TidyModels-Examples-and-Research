
###
## Chapter 6
###

# Setup
## Load necessary packages
load_p <- function(x, character.only = TRUE, ...){
  if(!require(x, character.only = character.only, ...))
    install.packages(x, ...)
  library(x, character.only = TRUE, ...)
}
load_p('tidymodels')
load_p('recipes')
load_p('modeldata') # loaded by tidymodels
load_p('dplyr')
load_p('purrr')
load_p('tidyr')
load_p('broom')

## Import the dataset
data(ames, package = 'modeldata')
ames


## 6.1
### Focus:
### Neighborhood
### general living area (Gr_Liv_Area)
### Year_Build
### building type (Bldg_Type)
set.seed(123) # reproducibility
ames_split <- ames %>% initial_split(0.8)

ames_subset <- ames_split %>%
  analysis() %>%
  select(Gr_Liv_Area,
         Year_Built,
         Bldg_Type,
         Neighborhood,
         Sale_Price)

# Create initial recipe replicating design matrix effect:
simple_ames <-
  recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type,
         data = ames_subset)  %>%
  step_log(Gr_Liv_Area, base = 10) %>%
  step_dummy(all_nominal())

simple_ames_no_formula <-
  recipe(ames_subset,
         roles = c(rep('predictor', 4), 'outcome')) %>%
  step_log(Gr_Liv_Area, base = 10) %>%
  step_dummy(all_nominal())

## Are they identical?
identical(simple_ames, simple_ames_no_formula)
## Hmm... why not.. I'm betting that it has something to do with with the formula..
simple_ames$term_info
simple_ames_no_formula$term_info
## yeah the term info is ordered differently, but otherwise are identical. Likely that is the problem.

# 6.2 Using recipes

## to use the recipe we need to prepare the steps using `prep`
prep_ames <- prep(simple_ames)
prep_ames_alt <- prep(simple_ames_no_formula)

### Note that the prepared data is stored in the dataset.
### Setting `retain = FALSE` can be used to change this behaviour
prep_ames_empty <- prep(simple_ames, retain = FALSE)
load_p('microbenchmark')
ames_train <- ames_split %>% analysis()
#### I wonder how much of an effect this has on the baking process
microbenchmark(juice(prep_ames),
                               bake(prep_ames,
                                    new_data = ames_train),
                               bake(prep_ames_empty,
                                    new_data = ames_train))
#### Interesting it doesn't seem to affect bake, only juice (which cant be used without retain)

### A bit further down we got to see that one could use `new_data = NULL` to return training result if it is stored.
microbenchmark(juice(prep_ames),
               bake(prep_ames, new_data = NULL))
### Well.. it should've worked. But it didn't


## To use the recipe we can use `bake(rec, new_data = ...)` or `juice(rec)`.
## For new data we always have to use `bake`

prep_ames %>% juice()
prep_ames %>% bake(new_data = ames_split %>% analysis())
prep_ames %>% bake(new_data = ames_split %>% assessment())

## We can also subset the columns of our output directly in bake
prep_ames %>% bake(new_data = ames_split %>% assessment(), starts_with('Neighborhood_'))
### I wonder if there are some optimizations for this
microbenchmark(bake = prep_ames %>% bake(new_data = ames_split %>% assessment(), starts_with('Neighborhood_')),
               bake_select = prep_ames %>% bake(new_data = ames_split %>% assessment()) %>% select(starts_with('Neighborhood_')))
### It does seem to be about 10 % faster. This might just be because of the compiled function however.

# 6.3: Encoding qualitative data in a numeric format.

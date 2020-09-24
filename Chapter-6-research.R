
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
load_p('ggplot2')
load_p('ggthemes')
load_p('patchwork')
## Import the dataset
data(ames, package = 'modeldata')
ames <- ames %>% mutate(Sale_Price = log10(Sale_Price))
ames


## Setup variables
graphics_dir <- 'chapter-6-graphics'

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
  step_log(Gr_Liv_Area, Sale_Price, base = 10) %>%
  step_dummy(all_nominal())

simple_ames_no_formula <-
  recipe(ames_subset,
         roles = c(rep('predictor', 4), 'outcome')) %>%
  step_log(Gr_Liv_Area, Sale_Price, base = 10) %>%
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

## For handling very small groups in nominal values we can use step_other while step_novel can be used for "expected future level"
## Neighborhood has an example here
ames_train %>%
  ggplot(aes(y = Neighborhood)) +
  geom_bar() +
  labs(y = NULL) +
  theme_pander()
ggsave(file.path(graphics_dir, 'Neighborhood.png'),
       width = 10, height = 8)
## Using step_other with threshold = 0.01 will ensure the smallest category is at least 1 % of the dataset

simple_ames <-
  recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type,
         data = ames_subset)  %>%
  step_log(Gr_Liv_Area, base = 10) %>%
  step_other(Neighborhood, threshold = 0.01) %>%
  prep()

### We can see that "other" has devoured many smaller groups.
simple_ames %>% juice() %>%
  ggplot(aes(y = Neighborhood)) +
  geom_bar() +
  labs(y = NULL) +
  theme_pander()
ggsave(file.path(graphics_dir, 'Neighborhood_grouped.png'),
       width = 10, height = 8)

## For ordered terms we can choose how we want to encode our dummy variables using specific steps.
## Eg.
mtcars %>%
  recipe( mpg ~ hp + cyl ) %>%
  step_integer(cyl) %>%
  step_num2factor(cyl, ordered = TRUE, levels = c(letters[1:3])) %>%
  prep() %>%
  juice()
## Note that there is a need for step_integer, although I've opened an issue at
## https://github.com/tidymodels/recipes/issues/575

# 6.4: Interaction terms

## As known by the most stupid beings, sometimes an effect can be different in different groupings or at different levels.
## For the ames dataset we have that for sale price given different living area across building type.
ggplot(ames_train, aes(x = Gr_Liv_Area, y = 10^Sale_Price)) +
  geom_point(alpha = .2) +
  facet_wrap(~ Bldg_Type) +
  geom_smooth(method = lm, formula = y ~ x, se = FALSE, col = "red") +
  scale_x_log10() +
  scale_y_log10() +
  labs(x = "General Living Area", y = "Sale Price (USD)") +
  theme_pander()
ggsave(file.path(graphics_dir, 'interaction_graphics.png'),
       width = 12, height = 8)
## We can use step_interact to add standard interactions.
## By default it will throw an error if a variable is not dummified. This is very unlike base R (I don't know if I like that yet.)
ames_train %>%
  recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type) %>%
  step_log(Gr_Liv_Area, base = 10) %>%
  step_other(Neighborhood, threshold = 0.01) %>%
  step_dummy(all_nominal()) %>%
  step_interact( ~ Gr_Liv_Area:starts_with("Bldg_Type_")) %>%
  prep() %>%
  juice() %>%
  names()
## Pretrained recipe
prep_ames %>%
  step_interact( ~ Gr_Liv_Area:starts_with("Bldg_Type_")) %>%
  prep() %>%
  juice() %>%
  names()


prep_ames %>%
  step_interact( ~ Gr_Liv_Area:starts_with("Bldg_Type_"), sep = ':') %>%
  prep() %>%
  juice() %>%
  names()


# 6.5: Skipping steps for new data




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
load_p('textrecipes')
load_p('tidymodels')
load_p('recipes')
load_p('modeldata') # loaded by tidymodels
load_p('dplyr')
load_p('purrr')
load_p('tidyr')
load_p('themis')
load_p('broom')
load_p('ggplot2')
load_p('ggthemes')
load_p('patchwork')
load_p('splines')
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

## Sometimes we're interested in skipping steps. Especially when it comes to imputing or subsamping data.
## For the test set we are not interested in the step being used upon the testing set
## to avoid this one can set "skip = TRUE" for steps in the formula
## Some steps such as "step_rose", "step_sample" do this by default.

### Create example or 2 here.
ames_train %>% mutate()

View(ames_train)
## Example with step_novel (add "high" as a level.) Does not skip on default
ames_train %>% select(Land_Contour) %>% summary()

ames_novel <- ames_train %>%
  recipe(Sale_Price ~ Gr_Liv_Area + Land_Contour) %>%
  step_log(Sale_Price, Gr_Liv_Area, base = 10) %>%
  step_novel(Land_Contour, new_level = 'Under water') %>%
  step_dummy(starts_with('Land_Contour')) %>%
  prep()
ames_test <- ames_split %>% assessment()
### Note that step_novel is kept throughout. So Under.water is kept as a level.
ames_novel %>% juice()
ames_novel %>% bake(new_data = ames_test)

## Example with... step_filter
ames_filter <- ames_train %>%
  recipe(Sale_Price ~ Gr_Liv_Area + Land_Contour) %>%
  step_log(Sale_Price, Gr_Liv_Area, base = 10) %>%
  step_filter(Land_Contour != 'Lvl') %>%
  prep()
### Note that Juice has no "Lvl" but bake returns the a tibble with observations in this field.
ames_filter %>% juice() %>% summary()
ames_filter %>% bake(new_data = ames_test) %>% summary()


# 6.6: Other examples of recipe steps

## Often when visualizing data we use a smoother to visualize the pattern of our data.
plot_smoother <- function(deg_free){
  ggplot(ames_train, aes(x = Latitude, y = Sale_Price)) +
    geom_point(alpha = 0.1) +
    scale_y_log10() +
    geom_smooth(method = lm,
                formula = y ~ ns(x, df=  deg_free),
                col = '#850030', # I like this colour better
                se = FALSE
                ) +
    ggtitle(paste(deg_free, 'Spline Terms'))
}
(plot_smoother(2) + plot_smoother(5)) /
   (plot_smoother(20) + plot_smoother(100))
ggsave(file.path(graphics_dir, 'smoother.png'))

## We can add this smoother to our data using step_ns
## Several other steps apparently exist? They are not mentioned, but i assume step_mutate

recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type + Latitude,
       data = ames_train) %>%
  step_log(Sale_Price, Gr_Liv_Area, base = 10) %>%
  step_other(Neighborhood, threshold = 0.01) %>%
  step_dummy(all_nominal()) %>%
  step_interact( ~ Gr_Liv_Area:starts_with("Bldg_Type_") ) %>%
  step_ns(Latitude, deg_free = 20) %>%
  prep() %>%
  juice() %>%
  select(starts_with('Latitude_')) %>%
  dim()
### Note that for deg_free 20 we of course have 20 polynomials across the data.

#- Feature extraction (subsection)
## We can similarly use step_pca for extracting uncorrelated features.
## It is very important to note that step_pca does NOT scale/center your variables
## So one has to add step_normalize beforehand.
recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type + Latitude,
       data = ames_train) %>%
  step_log(Sale_Price, Gr_Liv_Area, base = 10) %>%
  step_other(Neighborhood, threshold = 0.01) %>%
  step_dummy(all_nominal()) %>%
  step_normalize(matches('(SF$)|(GR_Liv)')) %>%
  step_pca(matches('(SF$)|(GR_Liv)')) %>%
  prep() %>%
  juice() %>%
  select(starts_with('PC'))


## One could also use "ICA", "KPCA", "NNMF", "MDS" or "UMAP"
?step_kpca

#- Row sampling steps (subsection)
## Both the recipes and themis package has some step for sampling data
## The themis package has most of the steps such as down- and upsampling, smote and so forth.

### There is no missing data in the ames dataset, so an example could be with the okc data
### See help(okc)
data(okc, package = 'modeldata')
summary(okc)

dim(okc) # Note 59855 by 6
okc %>% recipe(~ . ) %>%
  step_downsample(diet)  %>%
  prep() %>%
  juice() # Note 209 by 6 due to downsampling.

#### Downsmapling:
okc %>% select(diet) %>% table(useNA = 'always') # smallest group = 11, largest = 16562, 24360 NA.
okc %>% recipe(~ . ) %>%
  step_downsample(diet)  %>%
  prep() %>%
  juice() %>%
  select(diet) %>%
  table(useNA = 'always') # All groups = 11. Even NA
#### Upsampling
okc %>% recipe(~ . ) %>%
  step_upsample(diet)  %>%
  prep() %>%
  juice() %>%
  select(diet) %>%
  table(useNA = 'always') # All groups = 16562 but not NA (important difference from downsampling)


#- General transformation (subsection)
## Like dplyr we can use step_mutate and step_mutate_at to do transformations to the dataset
## Wonder if "across" works like dplyr too

# Using step_mutate instead of step_log (just.. for some example)
identical(recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type + Latitude,
                 data = ames_train) %>%
            step_log(Sale_Price, Gr_Liv_Area, base = 10) %>%
            prep() %>%
            juice(),
          recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type + Latitude,
                 data = ames_train) %>%
            step_mutate(Sale_Price = log(Sale_Price, base = 10), Gr_Liv_Area = log(Gr_Liv_Area, base = 10)) %>%
            prep() %>%
            juice()
)




#- Natural Language Processing
## Need an example here
## I have little experience.
?textrecipes
data(okc_text, package = 'modeldata')
okc_rec <- recipe(~  essay0 + essay1, data = okc_text) %>%
  step_tokenize(essay0, essay1) %>% # Tokenizes to words by default
  # Remove useless words such as "in" "a" "an" and so forth
  step_stopwords(essay0, essay1) %>% # Uses the english snowball list by default
  # Limit to the 100 most frequent tokens
  step_tokenfilter(essay0, essay1, max_tokens = 100) %>%
  step_tfidf(essay0, essay1)


# 6.7: How data are used by the recipe
## There is not really any examples here.
## Data is used depending on the step implemented and passed on for use in `prep`.
## Eg a step may need some information at the step phase and stores this info, while the `prep` step will use this information, unaltered by other steps
## Most steps however do "nothing" in the step phase and simply creates a specification (eg a recipe step) that should be performed by prep.
## that is why "step_*"s are fast while "prep"  is slow.
## Obviously in proper implementations only "predict"-type things are performed by "bake" and "juice".

# 6.8: Using a recipe with traditional modeling functions
## This section shows how we can use the final data in a standard modelling reference
ames_lm_rec <- recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type +
                            Latitude + Longitude, data = ames_train) %>%
  step_log(Sale_Price, Gr_Liv_Area, base = 10) %>%
  step_other(Neighborhood, threshold = 0.01) %>%
  step_dummy(all_nominal()) %>%
  step_interact( ~ Gr_Liv_Area:starts_with("Bldg_Type_") ) %>%
  step_ns(Latitude, Longitude, deg_free = 20) %>%
  prep()
ames_train_prepped <- ames_lm_rec %>% juice()
ames_test_prepped <- ames_lm_rec %>% bake(new_data = ames_test)

lm_ames_base <- lm(Sale_Price ~ ., ames_train_prepped)
glance(lm_ames_base)
tidy(lm_ames_base)

predict(lm_ames_base, data = ames_test_prepped)

# 6.9: Tidy af recipe
## Similar to brooming a fit we can use broom to get a tidy look into our recipe
tidy(ames_lm_rec)
## the id can be specified using `id` in the specific steps, and using the id we can see more information about the specific step
tidy(ames_lm_rec, id = 'log_fXqAl')
## We can also use the number of the step for more information
tidy(ames_lm_rec, number = 4)
## the output depends on the specific step one looks into, so it is always better to use "id" instead of number for further processing,
## in order to remove errors due to changed order or added steps.

# 6.10: Column roles
##

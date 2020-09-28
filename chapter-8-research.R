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

# 8.1: Where does the model begin and end?

## TidyModels use workflow to describe a process similar to a pipeline in Python, bash and so on. 
## In R a pipeline is usually used to describe the `%>%` operator from dplyr, as such workflow is used instead.
## In a classic modelling workframe the model itself is the entire workflow.
## the `workflows` package can be used to handle this problem.

# 8.2: Workflow basics


## The workflow object is a simple container for steps, actions and models.
## The steps and actions can be both 'pre' and 'post' processing.
## However for general usage the method is simply to use "workflow" and then "add_*"
lm_model <- linear_reg() %>%
  set_engine('lm')

lm_wf <- workflow() %>% 
  add_model(lm_model)

## The model visualizes model, pre- and postprocessing steps.
lm_wf

## We can add model input specification using `add_formula` or `add_recipe`
lm_wf <- lm_wf %>%
  add_formula(Sale_Price ~ Longitude + Latitude)
lm_wf

## Once specified we can use "fit" to fit the model process.
lm_f <- fit(lm_wf, ames_train_prep)

## We can similarly make predictions on the workflow
predict(lm_f, ames_test_prep %>% slice(1:6))


# 8.3: Workflows and recipes

## For pre-processing and model specification we could instead have used a recipe.
## But if we add a recipe to a workflow with a specifaction already,
## it will throw an error. We have to remove the formula first.

### Error example:
lm_wf %>% add_recipe(ames_rec)

### Removing the formula and adding recipe
lm_wf <- lm_wf %>% remove_formula() %>% 
  add_recipe(ames_rec)
### Note that the model is no longer fit when we add a recipe.
lm_f2 <- fit(lm_wf, ames_train)

## Ofcourse we can still predict our model, using the entire workflow
predict(lm_f2, ames_test %>% slice(1:3))

## If we want the innards of the object for example the prepped recipe we can use 
## pull_* 
## For example the individual steps in the workflow:
pull_workflow_fit(lm_f2)
pull_workflow_spec(lm_f2)
pull_workflow_preprocessor(lm_f2)
pull_workflow_prepped_recipe(lm_f2)

## Example: pulling the details of the workflow
pull_workflow_mold(lm_f2)

## This should note that the recipe object can get quite large quite quick, as it
## will contain not 1, but possibly multiple sets of the data. For example:
## lm_f2 has a data in the unprepped recipe
lm_f2$pre$actions$recipe$recipe$template
## but it also has the data in the fitted model (because this is standard in lm)
pull_workflow_fit(lm_f2)$fit$model
## This can make it rather large at times.

# 8.4: How does a workflow use the formula (`add_formula`)?

#- Tree-based models (subsection)
## The formula is used differently depending on the model interface.
## The implementation in `parsnip` was made with the knowledge of how different packages 
## accept input for the model. As such factors may or may not be converted to dummy variables 
## before being handed to the modelling function.


#- Special formulas and in-line functions (subsection)
## Some formulas are parsed in special ways by packages. One such example is random effects 
## The input formla to a workflow has to be consisting only of basic parts eg.
## the formula cant have mixed effects `(a | b)` as the term is unrecognized by 
## the standard formula interface. A workaround is that the `add_model`
## function lets one overwrite the formula passed to the actual function.
## Sadly there is no interface for mixed models as of yet.

parametric_model <- 
  surv_reg() %>% 
  set_engine("survival")

parametric_workflow <- 
  workflow() %>% 
  # This formula passes the data along as-is: 
  add_formula(futime + fustat ~ age + rx) %>% 
  add_model(parametric_model, 
            # This formula is given to the model
            formula = Surv(futime, fustat) ~ age + strata(rx))

random_effects <- fit(parametric_workflow, data = ovarian)
random_effects


# 8.5: Future plans
## They plan to extend the capacity of pre- and postprocesser, to take more customizable steps.
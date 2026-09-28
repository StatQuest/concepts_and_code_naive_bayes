# Load ggplot2 for visualization
library(ggplot2)

# Read the CSV file into a data frame using base R (no tidyverse)
df <- read.csv("diabetes_data.csv")

# Print the first few rows to inspect the data
head(df)

# Draw overlapping, semi-transparent histograms of glucose, colored by diabetes status
ggplot(df, aes(x = glucose, fill = factor(diabetes))) +
  # Use identity position so bars overlap rather than stack, with 50% transparency
  geom_histogram(position = "identity", alpha = 0.5, bins = 30)

# Draw overlapping, semi-transparent histograms of bloodpressure, colored by diabetes status
ggplot(df, aes(x = bloodpressure, fill = factor(diabetes))) +
  # Use identity position so bars overlap rather than stack, with 50% transparency
  geom_histogram(position = "identity", alpha = 0.5, bins = 30)

# Load tidymodels for data splitting utilities
library(tidymodels)

# Set a random seed so the split is reproducible
set.seed(3847)

# Create a stratified split, keeping 75% of rows for training, stratified on diabetes
split <- initial_split(df, prop = 0.75, strata = diabetes)

# Extract the training set from the split
train <- training(split)

# Extract the testing set from the split
test <- testing(split)

# Check class balance (counts and proportions) of diabetes in the training set
train |> count(diabetes) |> mutate(prop = n / sum(n))

# Check class balance (counts and proportions) of diabetes in the testing set
test |> count(diabetes) |> mutate(prop = n / sum(n))

# Load the naivebayes package for fitting the classifier
library(naivebayes)

# Fit a Gaussian naive Bayes classifier predicting diabetes from glucose and bloodpressure
nb_model <- naive_bayes(factor(diabetes) ~ glucose + bloodpressure, data = train)

# Print a summary of the fitted model (priors, per-class means/sds for each predictor)
nb_model

# Generate class predictions for the test set using the fitted model
test_pred <- predict(nb_model, newdata = test)

# Store the predictions as a new column on the test data frame
test$pred <- test_pred

# Build a confusion matrix comparing true diabetes labels to predicted labels
conf_mat <- table(truth = test$diabetes, predicted = test_pred)

# Print the confusion matrix
conf_mat

# Compute overall accuracy as the proportion of correct predictions
accuracy <- mean(test_pred == test$diabetes)

# Print the accuracy
accuracy

###
### See how to classify a new patient!!!
###

# Create a new data frame representing a single new patient with glucose and bloodpressure values
new_patient <- data.frame(glucose = 50, bloodpressure = 75)

# Predict the class label (0 = no diabetes, 1 = diabetes) for the new patient
predict(nb_model, newdata = new_patient)

# Predict class probabilities for the new patient instead of just the class label
predict(nb_model, newdata = new_patient, type = "prob")


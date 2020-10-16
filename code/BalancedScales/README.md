# Balanced Scale Classification
by left and right weights and distance offsets

Article Tutorial [here](https://ehackz.com/2018/03/23python-scikit-learn-random-forest-classifier-tutorial/)

## Data Descriptions
* 4 measurements are included in the data file:
  left_weight, left_distance, right_weight, right_distance

* 3 balance classifications:
  balanced, left Skewed, right Skewed

## Feature Engineering
* 3 additional parameters are calculated and added.

1. (x2) The cross products of side weight and offset distance\
Calculates the side weight times the offset distance to derive a relationship between the distance away and how heavy the side is

2. The ratio of cross products
Calculates the ratio between the left and right cross products to derive a direct relationship between the sides. Impact can be seen in the final model analysis.

## Hyperparameter Tuning
* Three hyperparameters are used to fine tune the accuracy of the random forest model using GridSearchCV
1. n_estimators\
The number of decision trees to use for our random forest model
2. max_depth\
The maximum depth of each decision tree
3. min_samples_leaf\
The minimum number of samples required to be at a leaf node in each decision tree
from code.Utilities.utils import csv_retrieve_dataset, get_input
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plot
import pandas
import os, sys, json

class BalancedScales():
  """
    A class designed to train, test, and explore with a Random Forest Classifier
    machine learning model
  """

  # Data location constants
  TRAINING_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data"
  TRAINING_DATA_PATH = "code/BalancedScales/data/scale_raw.csv"

  # Data description constants
  COLUMN_NAMES = ["tilt","left_weight","left_distance","right_weight", "right_distance"]

  dataset = None
  model = None
  data_columns = None
  target_columns = None

  def __init__(self):
    """
    Set up some basic class variables and load the needed data
    """
    # Retrieve the training dataset if needed
    if(not os.path.exists(self.TRAINING_DATA_PATH)):
      print("Training data file does not exists, retrieving")
      self.load_dataset()
    else:
      print("Training data file exists, moving on")

    # Load the training dataset into memory
    if (self.dataset is None):
      print("Reading dataset")
      self.set_dataset(pandas.read_csv(self.TRAINING_DATA_PATH))

    # isolate the data columns and target columns
    self.set_data_columns(self.COLUMN_NAMES[1:])
    self.set_target_columns(self.COLUMN_NAMES[:1])

  def set_dataset(self,dataset):
    """ Class variable setter """
    self.dataset = dataset

  def set_model(self,model):
    """ Class variable setter """
    self.model = model

  def set_accuracy(self,accuracy):
    """ Class variable setter """
    self.accuracy = accuracy

  def set_data_columns(self, columns):
    """ Class variable setter """
    self.data_columns = columns

  def set_target_columns(self, columns):
    """ Class variable setter """
    self.target_columns = columns

  def load_dataset(self):
    """ Retrieve the data from an external URL"""
    print("Loading Dataset")
    csv_retrieve_dataset(self.TRAINING_DATA_PATH,self.TRAINING_DATA_URL,self.COLUMN_NAMES)

  def analysis(self):
    """
      Provide and display some basic analysis about the dataset
    """
    print(self.dataset.head())
    print(self.dataset.info())
    print(self.dataset[self.COLUMN_NAMES[0]].value_counts())

    pandas.plotting.scatter_matrix(self.dataset)
    plot.show()

  def split_data(self):
    """
      Convert the raw dataset into a training and testing set to be used by the model
    """
    # add on some additional column headers here if they are included by the feature engineering process
    data_columns = self.data_columns
    target_columns = self.target_columns

    X = self.dataset[data_columns]
    y = self.dataset[target_columns]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return (X_train, X_test, y_train, y_test)

  def feature_engineering(self, with_analysis = False):
    """
      Add on some additional column headers to the dataset to increase the model accuracy
    """
    # step one of feature engineering
    self.dataset['left_cross'] = self.dataset['left_distance'] * self.dataset["left_weight"]
    self.dataset['right_cross'] = self.dataset['right_distance'] * self.dataset["right_weight"]
    # step two of feature engineering
    self.dataset["left_right_ratio"] = self.dataset["left_cross"]/self.dataset["right_cross"]

    if with_analysis:
      self.analysis()

    self.set_data_columns(self.data_columns + ['left_cross','right_cross','left_right_ratio'])

  def train_model(self, with_analysis, hyperparameters = False):
    """
      Split the dataset,
      create a Random forest classifier model,
      train with part of the dataset,
      and test with the final portion of data
    """
    print("Training Model")

    self.feature_engineering(with_analysis)

    X_train, X_test, y_train, y_test = self.split_data()

    # Create and fit the model
    forest = RandomForestClassifier()

    if (hyperparameters):
      params = {
        "n_estimators": [100, 300, 500],
        "max_depth": [5,8,15],
        "min_samples_leaf" : [1, 2, 4]
      }
      forest = GridSearchCV(forest, param_grid=params, cv=5 )

    forest.fit(X_train, y_train.values.ravel())
    
    # Test the results
    y_predict = forest.predict(X_test)
    accuracy = accuracy_score(y_test.values, y_predict)
    print("Model Accuracy: " + str(accuracy))
    self.set_model(forest)
    self.set_accuracy(accuracy)

  def model_analysis(self, hyperparameters = False):
    if (not hyperparameters):
      features_dict = {}
      for index in range(len(self.model.feature_importances_)):
        features_dict[self.data_columns[index]] = self.model.feature_importances_[index]
      print("Feature importance")
      print(json.dumps(sorted(features_dict.items(), key=lambda x:x[1], reverse=True),indent=4))
    else:
      print(self.model.best_params_)
      print(self.model.best_score_)

  def test_model(self,test_data = None):
    if test_data is None:
      print()
      print("Welcome to the balancing scale model")
      print("Now that the model is trained, enter in some values to get its prediction")
      
      test_data = []
      test_data.append(get_input("Left Weight: ",int))
      test_data.append(get_input("Left Distance: ",int))
      test_data.append(get_input("Right Weight: ",int))
      test_data.append(get_input("Right Distance: ",int))
      test_data.append(test_data[0] * test_data[1])
      test_data.append(test_data[2] * test_data[3])
      test_data.append(test_data[4]/test_data[5])

      print("[Balanced, Left Skewed, Right Skewed] liklihood probabilities")
      print(self.model.predict_proba([test_data]))
      print("Did it get it right? If not, that's why the accuracy is " + str(self.accuracy))
      while True:
        answer = input("Try again? (y/n): ")
        if (answer.lower() == 'y' or answer.lower() == 'yes'):
          return self.test_model()
        if (answer.lower() == 'n' or answer.lower() == 'no'):
          break
      return


if __name__ == "__main__":
    balanced_scales = BalancedScales()
    # balanced_scales.dataset_analysis()
    balanced_scales.train_model(with_analysis = False, hyperparameters=True)
    balanced_scales.model_analysis(hyperparameters=True)
    balanced_scales.test_model()
    
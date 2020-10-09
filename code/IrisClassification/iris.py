import pandas
import numpy
import requests
import seaborn
import matplotlib.pyplot as plot
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer, normalize
from sklearn.model_selection import train_test_split
from datetime import datetime

import os, sys

class IrisSpeciesPredictor:
  """ A class used to predict the species of a iris by its measurements using a Neural Network trained with Supervised learing"""

  # Data Location Constants
  TRAINING_DATA_PATH = "data/iris_raw.csv"
  TRAINING_DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
  TESTING_DATA_PATH = "data/iris_test.csv"
  TESTING_DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"

  # Data Description Contants
  CLASS_NAMES = ['Iris setosa', 'Iris versicolor', 'Iris virginica']
  DATA_HEADERS = ["sepal_length","sepal_width","petal_length","petal_width","species"]
  FEATURE_NAMES = DATA_HEADERS[:-1]
  LABEL_NAMES = DATA_HEADERS[-1]
  BATCH_SIZE = 32

  # Model Variables
  training_dataset = None
  testing_dataset = None
  model = None
  loss_object = None
  optimizer = None

  def __init__(self):
    # Change the current working directory to the location of the running file
    os.chdir(sys.path[0])

    # Retrieve the training dataset if needed
    if(not os.path.exists(self.TRAINING_DATA_PATH)):
      print("Training data file does not exists, retrieving")
      self.retrieve_dataset(self.TRAINING_DATA_PATH, self.TRAINING_DATA_URL)
    else:
      print("Training data file exists, moving on")

    # Load the training dataset into memory
    if (self.training_dataset is None):
      self.set_training_dataset(tf.data.experimental.make_csv_dataset(
        self.TRAINING_DATA_PATH,
        self.BATCH_SIZE,
        column_names=self.DATA_HEADERS,
        label_name=self.LABEL_NAMES,
        num_epochs=1
      ))

    # Retreive the testing dataset if needed
    if(not os.path.exists(self.TESTING_DATA_PATH)):
      print("Testing data file does not exists, retrieving")
      self.retrieve_dataset(self.TESTING_DATA_PATH, self.TESTING_DATA_URL)
    else:
      print("Testing data file exists, moving on")

    # Load the testing dataset into memory
    if (self.testing_dataset is None):
      self.set_testing_dataset(tf.data.experimental.make_csv_dataset(
        self.TESTING_DATA_PATH,
        self.BATCH_SIZE,
        column_names=self.DATA_HEADERS,
        label_name=self.LABEL_NAMES,
        num_epochs=1,
        shuffle=False
      ))

  def set_training_dataset(self,dataset):
    self.training_dataset = dataset
  
  def set_testing_dataset(self,dataset):
    self.testing_dataset = dataset

  def set_model_definition(self,definition):
    self.model = definition

  def set_loss_object(self,loss_object):
    self.loss_object = loss_object

  def set_optimizer(self,optimizer):
    self.optimizer = optimizer

  def retrieve_dataset(self, output_file_path, data_url):
    """ A function to retrieve data files from external urls.
        Moves the downloaded file to the current directory,
        normally which would be stored in the tensorflow installation location. """
    tf_file_pointer = tf.keras.utils.get_file(fname=os.path.basename(data_url),origin=data_url)

    with open(tf_file_pointer,"r") as input_file:
      with open(output_file_path,"a") as output_file:
        line = input_file.readline()
        while line:
          output_file.write(line)
          line = input_file.readline()
      output_file.close()
    input_file.close()

  def describe_dataset(self):
    print("Features: {}".format(self.FEATURE_NAMES))
    print("Label: {}".format(self.LABEL_NAMES))

  def visualize_dataset(self):
    features, labels = next(iter(self.training_dataset))
    # Plot the dataset
    plot.scatter(features['petal_length'],
              features['sepal_length'],
              c=labels,
              cmap='viridis')
    plot.xlabel("Petal length")
    plot.ylabel("Sepal length")
    plot.show()

  def pack_features_vector(self, features, labels):
    """Pack the features into a single array."""
    features = tf.stack(list(features.values()), axis=1)
    return features, labels

  def translate_dataset(self,dataset, training):
    if (training):
      self.set_training_dataset(dataset.map(self.pack_features_vector))
    else:
      self.set_testing_dataset(dataset.map(self.pack_features_vector))

  def build_model(self):
    self.set_model_definition(tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),
        tf.keras.layers.Dense(10, activation=tf.nn.relu),
        tf.keras.layers.Dense(3)
      ])
    )

  def build_loss_object(self):
    self.set_loss_object(tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

  def build_optimizer(self):
    self.set_optimizer(tf.keras.optimizers.SGD(learning_rate=0.01))

  def calculate_gradient(self, inputs, targets):
    with tf.GradientTape() as tape:
      loss_value = self.calculate_loss(inputs, targets, training=True)
    return loss_value, tape.gradient(loss_value, self.model.trainable_variables)

  def calculate_loss(self, x, y, training):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    y_hat = self.model(x, training=training)

    return self.loss_object(y_true=y, y_pred=y_hat)

  def train_model(self, num_epochs = 201):

    # Build the model and necessary objects
    if (self.model is None):
      self.build_model()
    if (self.loss_object is None):
      self.build_loss_object()
    if (self.optimizer is None):
      self.build_optimizer()

    self.translate_dataset(self.training_dataset,True)
  
    # Keep results for plotting
    train_loss_results = []
    train_accuracy_results = []

    for epoch in range(num_epochs):
      epoch_loss_avg = tf.keras.metrics.Mean()
      epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

      # Training loop - using batches of 32
      for x, y in self.training_dataset:
        # Optimize the model
        loss_value, grads = self.calculate_gradient(x, y)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # Track progress
        epoch_loss_avg.update_state(loss_value)  # Add current batch loss
        # Compare predicted label to actual label
        # training=True is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        epoch_accuracy.update_state(y, self.model(x, training=True))

      # End epoch
      train_loss_results.append(epoch_loss_avg.result())
      train_accuracy_results.append(epoch_accuracy.result())

      if epoch % 25 == 0:
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch, epoch_loss_avg.result(), epoch_accuracy.result()))

    return (train_loss_results,train_accuracy_results)

  def training_details(self, loss, accuracy):
    fig, axes = plot.subplots(2, sharex=True, figsize=(12, 8))
    fig.suptitle('Training Metrics')

    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].plot(loss)

    axes[1].set_ylabel("Accuracy", fontsize=14)
    axes[1].set_xlabel("Epoch", fontsize=14)
    axes[1].plot(accuracy)
    plot.show()

  def test_model(self, test_score=False):
    self.translate_dataset(self.testing_dataset,False)

    test_accuracy = tf.keras.metrics.Accuracy()

    for (x, y) in self.testing_dataset:
      # training=False is needed only if there are layers with different
      # behavior during training versus inference (e.g. Dropout).
      logits = self.model(x, training=False)
      prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
      test_accuracy(prediction, y)

    print("Test set accuracy: {:.3%}".format(test_accuracy.result()))
    if (test_score):
      print(tf.stack([y,prediction],axis=1))


  def predict_model(self):
    predict_dataset = tf.convert_to_tensor([
      [5.1, 3.3, 1.7, 0.5,],
      [5.9, 3.0, 4.2, 1.5,],
      [6.9, 3.1, 5.4, 2.1]
    ])

    # training=False is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = self.model(predict_dataset, training=False)

    for i, logits in enumerate(predictions):
      class_idx = tf.argmax(logits).numpy()
      p = tf.nn.softmax(logits)[class_idx]
      name = self.CLASS_NAMES[class_idx]
      print("Example {} prediction: {} ({:4.1f}%)".format(i, name, 100*p))
  

if __name__ == "__main__":
    irisSpeciesPredictor = IrisSpeciesPredictor()
    
    # Details
    irisSpeciesPredictor.describe_dataset()
    # irisSpeciesPredictor.visualize_dataset()

    # Training
    (loss, accuracy) = irisSpeciesPredictor.train_model()
    irisSpeciesPredictor.training_details(loss, accuracy)

    # Testing
    irisSpeciesPredictor.test_model(test_score=True)

    # Perdiction
    irisSpeciesPredictor.predict_model()


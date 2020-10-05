import pandas
import numpy
import matplotlib.pyplot as plot
import seaborn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error,mean_absolute_error

import os,sys,json

from visualizations import analysis, visualize

def main():
  # Change the current working directory to the location of the running file
  os.chdir(sys.path[0])

  # Import and view the housing data
  data_frame = pandas.read_csv("../data/california_housing_1990.csv")

  # show details of and clean up the data
  data_frame = analysis(data_frame)
  # show some basic visualizations about the data
  # visualize(data_frame)

  # break up the testing and training sets
  X = data_frame.drop("median_house_value",axis=1)
  y = data_frame["median_house_value"]

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

  scaler = MinMaxScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)

  model = Sequential()
  model.add(Dense(8,activation="relu"))
  model.add(Dropout(0.5))

  model.add(Dense(1))

  model.compile(optimizer="adam",loss="mse")

  early_stop = EarlyStopping(monitor="val_loss",mode="min", verbose=1,patience=10)

  model.fit(x=X_train,y=y_train.values,validation_data=(X_test,y_test.values),batch_size=128, epochs=400, callbacks=[early_stop])

  losses = pandas.DataFrame(model.history.history)
  losses.plot()
  plot.show()

  predictions = model.predict(X_test)

  print(mean_absolute_error(y_test,predictions))
  print(numpy.sqrt(mean_squared_error(y_test,predictions)))


if __name__ == "__main__":
    main()
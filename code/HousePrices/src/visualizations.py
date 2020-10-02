import pandas
import numpy
import matplotlib.pyplot as plot
import seaborn
import os,sys,json

def analysis(data_frame):
  print("\nPeak the head of the data frame")
  print(data_frame.head())

  print("\nBreak down information about the data frame")
  print(data_frame.info())

  # view any null values
  print("\nView bad values in the data")
  print(data_frame.isnull().sum())

  # remove the null values, drop the oceanProximity column
  data_frame.dropna(inplace=True)
  data_frame = data_frame.drop("ocean_proximity", axis = 1)
  print("\nView the new columns in the data frame")
  print(data_frame.columns)
  print("\n")
  print(data_frame.describe())
  return data_frame


def visualize(data_frame):
  # show the distribution of the median house values
  plot.figure(figsize=(12,8))
  seaborn.distplot(data_frame["median_house_value"])
  plot.title("Distribution of Median House Value")
  plot.show()

  # show the breakdown of median house values by longitude
  plot.figure(figsize=(12,8))
  seaborn.scatterplot(x='median_house_value',y='longitude',data=data_frame)
  plot.title("Median House Value by Longitude")
  plot.show()

  # show the breakdown of median house values by latitude
  plot.figure(figsize=(12,8))
  seaborn.scatterplot(x='median_house_value',y='latitude',data=data_frame)
  plot.title("Median House Value by Latitude")
  plot.show()

  # Showing latitude vs longitude with a colour gradient showing median house value
  plot.figure(figsize=(12,8))
  seaborn.scatterplot(x="longitude",y="latitude",data=data_frame,hue='median_house_value')
  plot.title("Longitude v. Latitude, with Median House Value")
  plot.show()

  # Show the correlation between each paired parameter in the data frame
  print(data_frame.corr())

  # Visually show the corrilation between parameter pairs with a heatmap
  plot.figure(figsize=(12,8))
  seaborn.heatmap(data_frame.corr(),annot=True)
  plot.title("")
  plot.show()
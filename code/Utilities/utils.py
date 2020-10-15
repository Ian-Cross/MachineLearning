import tensorflow as tf
import pandas
import os, csv, json
import requests

def tf_retrieve_dataset(output_file_path, data_url):
  """ A function to retrieve data files from external urls formatted as tensorflow files .
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

def csv_retrieve_dataset(output_file_path, data_url, names):
  """ A function to retrieve csv data files from an external urls
      Saves a local copy in the provided location """
  with requests.Session() as s:
    download = s.get(data_url)

    decoded_content = download.content.decode('utf-8')
    cr = csv.reader(decoded_content.splitlines(), delimiter=',')
    scales_data = list(cr)
    scales_data.insert(0,names)

    with open(output_file_path, 'w', newline='') as csvfile:
      csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
      csv_writer.writerows(scales_data)


def get_input(message, input_type):
  user_input = input(message)

  try:
    user_input = input_type(user_input)
    return user_input
  except ValueError:
    print("Input type required: " + str(input_type))
    print("User input type: " + str(type(user_input)))
    return get_input(message,input_type)
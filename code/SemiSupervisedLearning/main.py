from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def create_dataset():
  X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, random_state=1)
  return (X,y)

def split_data(X,y):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=1, stratify=y)
  return (X_train, X_test, y_train, y_test)




def main():
  # Create and Split the dataset
  (X, y) = create_dataset()
  (X_train, X_test, y_train, y_test) = split_data(X,y)

  # Split the data again into labeled and "unlabeled" pieces
  X_train_lab, X_test_unlab, y_train_lab, y_test_unlab = train_test_split(X_train, y_train, test_size=0.50, random_state=1, stratify=y_train)

  # Summarize Data
  print('Labeled Train Set:', X_train_lab.shape, y_train_lab.shape)
  print('Unlabeled Train Set:', X_test_unlab.shape, y_test_unlab.shape)
  print('Test Set:', X_test.shape, y_test.shape)

if __name__ == "__main__":
  main()
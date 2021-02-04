import nltk
import random
from nltk.corpus import movie_reviews, stopwords


def main():
    documents = [(list(movie_reviews.words(fileid)), category)
                 for category in movie_reviews.categories()
                 for fileid in movie_reviews.fileids(category)]

    random.shuffle(documents)

    all_words = []
    stop_words = stopwords.words('english')
    punctuation = [".", ",", "!", "'", '"', "-", ")",
                   "(", ":", "/", "?", ";", "[", "]", "{", "}", "@", "#", "$", "%", "^", "&", "*", "_"]
    for word in movie_reviews.words():
        if word not in stop_words and word not in punctuation:
            all_words.append(word.lower())

    all_words = nltk.FreqDist(all_words)

    # print(all_words.most_common(15))
    # all_words.plot(30, cumulative=False)

    word_features = list(all_words.keys())[:3000]

    feature_sets = [(find_features(rev, word_features), category)
                    for (rev, category) in documents]

    training_set = feature_sets[:1900]
    testing_set = feature_sets[1900:]

    classifier = nltk.NaiveBayesClassifier.train(training_set)
    print("Naive Bayes Accuracy Precentage: ",
          (nltk.classify.accuracy(classifier, testing_set))*100)
    classifier.show_most_informative_features(15)


def find_features(document, word_features):
    words = set(document)
    features = {}
    for word in word_features:
        features[word] = (word in words)
    return features


if __name__ == "__main__":
    main()

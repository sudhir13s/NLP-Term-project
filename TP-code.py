"""
Created on Sat May 5 01:12:48 2019

@author: sudhirsingh

This is the NLP term project code for identification of similar languages and varieties.
"""

import sys
# import nltk
import string

# from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import RandomOverSampler
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

# from sklearn.utils.multiclass import unique_labels
from sklearn import metrics
# from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def read_file_and_preprocess(file_name):
    """
    Function to read file and perform pre-processing.

    :param file_name: file to read and perform pre-processing.
    :return: na
    """
    data = open(file_name).read()
    texts, labels = [], []
    for idx, line in enumerate(data.split("\n")):
        line_contents = line.split('\t')
        if any(line_contents):
            line = line_contents[0]
            line = line.replace(".", "")
            line = line.translate(str.maketrans('','','1234567890'))
            line = line.translate(str.maketrans('', '', string.punctuation))
            line = line.translate(str.maketrans('', '', '#NE#'))
            line = line.replace('"', '')
            line = line.replace('\n', '')
            line = line.strip()
            texts.append(line)
            labels.append(line_contents[1])
    return texts, labels


def unique_class_label(y_train):
    """
    Function to retrieve unique class labels from test data set.
    :param y_train: all class labels from test data set.
    :return: list of unique class labels.
    """
    return list(set(y_train))


def print_line_each_language(X, y):
    """
    Function to print each line of different language.
    :param X: all sentences from dev & test data set.
    :param y: all calss lables from dev & test data set.
    :return: na
    """
    unique_lang = set()
    for line, l_code in zip(X, y):
        if l_code not in unique_lang:
            unique_lang.add(l_code)
            print(line + "\t" + l_code)


def feature_extraction_and_modeling(X_train, X_test, feature="tf-idf", ngram=1):
    """
    Function to create and extract features from dev and test data set.
    :param X_train: the train data set.
    :param X_test: the test data set.
    :param feature: the feature to be used.
    :param ngram: the n-grams to be used in the feature selection.
    :return: train and test feature matrix
    """
    #     feature_type = feature
    if feature == "tf-idf":
        # ngram level tf-idf
        # min_df=0.01, max_df = 0.95,
        # create count vector object
        word_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(1, ngram), max_df=0.95, max_features=None)
        # fit (create matrix) & transform the count vector object according to data
        word_vectorizer_X_train = word_vectorizer.fit_transform(X_train)
        # create tf-idf transformer
        tf_idf_transformer = TfidfTransformer()
        # transform the fitted word vector to tf-idf matrix
        X_train_feature_matrix = tf_idf_transformer.fit_transform(word_vectorizer_X_train)

        # do the same for test.
        word_vectorizer_X_test = word_vectorizer.transform(X_test)
        tf_idf_transformer = TfidfTransformer()
        X_test_feature_matrix = tf_idf_transformer.fit_transform(word_vectorizer_X_test)

        return X_train_feature_matrix, X_test_feature_matrix
    else:
        # create count vector object
        count_vector = CountVectorizer(analyzer='word', ngram_range=(1, ngram), max_df=0.95, max_features=None)
        # fit (create matrix) the count vector object according to data
        count_vector.fit(X_train)
        # transform data using count vector object
        X_train_feature_matrix = count_vector.transform(X_train)
        # do the same for test.
        X_test_feature_matrix = count_vector.transform(X_test)

        return X_train_feature_matrix, X_test_feature_matrix



def train_model_naive_bayes(X, y):
    """
    The Naive Bayes classifier model.
    :param X: the train data set.
    :param y: the test data set.
    :return: the Naive Bayes classifier model
    """
    ros = RandomOverSampler(random_state=None)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    nb_classifier = MultinomialNB(alpha=1.0, fit_prior=True)
    nb_classifier.fit(X_resampled, y_resampled)
    return nb_classifier


def train_model_LinearSVC(X, y):
    """
    The Linear SVM classifier model.
    :param X: the train data set.
    :param y: the test data set.
    :return: the Linear SVM classifier model
    """
    ros = RandomOverSampler(random_state=None)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    svm_classifier = LinearSVC(random_state=None)
    svm_classifier.fit(X_resampled, y_resampled)
    return svm_classifier


def train_model_LogisticRegression(X, y):
    """
    The Logistic regression classifier model.
    :param X: the train data set.
    :param y: the test data set.
    :return: the Logistic regression classifier model
    """
    ros = RandomOverSampler(random_state=None)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    lr_classifier = LogisticRegression(n_jobs=1, random_state=0, C=1e5, solver='lbfgs', max_iter=10000, multi_class='multinomial')
    lr_classifier.fit(X_resampled, y_resampled)
    return lr_classifier


def test_data_with_model(model, X, y):
    """
    Function to predict the test labels using different models.
    :param model: models to be used for this test.
    :param X: the train data set.
    :param y: the test data set.
    :return: the prediction and accuracy of the model provided on test data set.
    """
    prediction = model.predict(X)
    accuracy = model.score(X, y)
    return prediction, accuracy


def matrix_and_report(y_test, y_pred, unique_class):
    """
    Function to print confusion matrix and classification report on the screen.
    :param y_test: the test data set (actual class labels).
    :param y_pred: the predicted result on test data set (predicted class labels).
    :param unique_class: the unique class labels for the test data set.
    :return: na
    """
    tf_idf_ngram_cm = metrics.confusion_matrix(y_test, y_pred, labels=unique_class)
    figure, axis = plt.subplots(figsize=(10,10))
    axis.set_title('Confusion matrix')
    sns.heatmap(tf_idf_ngram_cm, annot=True, fmt='d', xticklabels=unique_class, yticklabels=unique_class)
    plt.ylabel('Predicted')
    plt.xlabel('Actual')
    plt.show()
    print("--------------------------------------------------------------------------")
    print("Confusion matrix, without normalization")
    print(metrics.confusion_matrix(y_test, y_pred, labels=unique_class))
    plt.show()
    print("--------------------------------------------------------------------------")
    print("Classification Report")
    print(metrics.classification_report(y_test, y_pred, labels=unique_class))
    print("--------------------------------------------------------------------------")


def perform_operations(file_train, file_dev_test, file_test):
    """
    The main function to perform operations on train and test data set.
    operation performed:
        - read file and pre-processing
        - extract unique class labels
        - feature extraction and modeling
        - model creation, Naive Bayes & Linear SVM
        - model prediction and accuracy
        - print model accuracy
        - print confusion matrix & classification report on screen.

    :param file_train: the train data set file name
    :param file_dev_test: the development test data set file name
    :param file_test: the test data set file name
    :return: na
    """
    """
    Training and testing on dev data
    """
    # Read file and preprocess the file
    X_train, y_train = read_file_and_preprocess(file_train)
    X_dev_test, y_dev_test = read_file_and_preprocess(file_dev_test)
    # Get all unique class labels (unique languages)
    train_unique_class = unique_class_label(y_train)
    # # Feature selection and modeling for train data
    X_train_tf_idf_ngram, X_dev_test_tf_idf_ngram = feature_extraction_and_modeling(X_train, X_dev_test,
                                                                                    feature="tf-idf", ngram=9)
    # Using Naive Bayes: create model
    nb_model = train_model_naive_bayes(X_train_tf_idf_ngram, y_train)
    # Using Naive Bayes: Test and make prediction of train data
    nb_prediction, nb_accuracy = test_data_with_model(nb_model, X_dev_test_tf_idf_ngram, y_dev_test)
    # Using Naive Bayes: Accuracy of predicted dev test data
    print("Multinomial Naive Bayes classifier accuracy score for test set=%0.4f" % nb_accuracy)
    # Multinomial Naive Bayes - Confusion matrix and classification report of train data
    matrix_and_report(y_dev_test, nb_prediction, train_unique_class)

    # Using Linear SVM: create model
    svm_model = train_model_LinearSVC(X_train_tf_idf_ngram, y_train)
    # Using Linear SVM: Test and make prediction of train data
    svm_prediction, svm_accuracy = test_data_with_model(svm_model, X_dev_test_tf_idf_ngram, y_dev_test)
    # Using Linear SVM: Accuracy of predicted dev test data
    print("Linear SVM classifier accuracy score for test set=%0.4f" % svm_accuracy)
    # Linear SVM - Confusion matrix and classification report of train data
    matrix_and_report(y_dev_test, svm_prediction, train_unique_class)

    """
    Testing on test data
    """
    # Read file and preprocess the file
    test_texts, test_classes = read_file_and_preprocess(file_test)
    # Get all unique class labels (unique languages)
    test_unique_class = unique_class_label(test_classes)
    # Feature selection and modeling for test data.
    X_train_tf_idf_ngram, test_tf_idf_ngram = feature_extraction_and_modeling(X_train, test_texts, feature="tf-idf",
                                                                              ngram=9)
    # Using Naive Bayes: Test and make prediction of test data
    nb_test_prediction, nb_test_accuracy = test_data_with_model(nb_model, test_tf_idf_ngram, test_classes)
    # Using Naive Bayes: Accuracy of predicted test data
    print("Multinomial Naive Bayes classifier accuracy score for test set=%0.4f" % nb_accuracy)
    # Using Naive Bayes: Print Confusion matrix and classification report of test data
    matrix_and_report(test_classes, nb_test_prediction, test_unique_class)

    # Using Linear SVM: Test and make prediction of test data
    svm_test_prediction, svm_test_accuracy = test_data_with_model(svm_model, test_tf_idf_ngram, test_classes)
    # Using Linear SVM: Accuracy of predicted test data
    print("Linear SVM classifier accuracy score for test set=%0.4f" % svm_test_accuracy)
    # Using Linear SVM: Print Confusion matrix and classification report of test data
    matrix_and_report(test_classes, svm_test_prediction, test_unique_class)


def main():
    """
    The main function to read file name and call perform_operations function for further processing.
    :return: na
    """
    if len(sys.argv) == 4:
        file_train = sys.argv[1]
        file_dev_test = sys.argv[2]
        file_test = sys.argv[3]
    else:
        file_train = input("Enter train file name:")
        file_dev_test = input("Enter dev test file name:")
        file_test = input("Enter test file name:")

    if file_train != '' and file_dev_test != '' and file_test != '':
        perform_operations(file_train, file_dev_test, file_test)
    else:
        print("Empty file name(s).")


if __name__=='__main__':
    main()

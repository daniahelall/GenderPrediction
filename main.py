import pandas
# Import train_test_split function
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
# Import scikit-learn metrics module for accuracy calculation
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

# reading the data from csv file
female_names = pandas.read_csv('female_names.csv', header=0)
male_names = pandas.read_csv('male_names.csv', header=0)

# Assign the value 0 to female and 1 to male
female_names['gender'] = 0
male_names['gender'] = 1

# We join the datasets and eliminate possible duplicate data

data = female_names._append(male_names, ignore_index=True)
data = data.drop_duplicates(subset='name', keep=False)

label = data['gender'].astype(str)

del (data['frequency'])
del (data['mean_age'])
del (data['gender'])

features = data['name'].astype(str)

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=7)

#####################################################################
# SVM Classifier

svm_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 1))),
                    ('tfidf', TfidfTransformer(norm='l2', sublinear_tf=True, use_idf=False)),
                    ('clf', SVC(kernel='linear'))])

svm_clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = svm_clf.predict(X_test)

# Model Accuracy: how often is the classifier correct?
print("SupportVectorMachine Classifier Accuracy: ", metrics.accuracy_score(y_test, y_pred))
print("---------------------------------------")
# print("Predictions: ")
# print(svm_clf.predict(("LUCIA",)))
# print(svm_clf.predict(("BORJA",)))
# print(svm_clf.predict(("MATEO",)))
# print(svm_clf.predict(("VICTORIA",)))
# print(svm_clf.predict(("FEDERICO",)))
# print(svm_clf.predict(("DANIELA",)))
# print(svm_clf.predict(("ALEJANDRA",)))
# print(svm_clf.predict(("SAID",)))

#####################################################################
# Naive Bayis classifier
NB_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 1))),
                   ('tfidf', TfidfTransformer(norm='l2', sublinear_tf=True, use_idf=False)),
                   ('clf', MultinomialNB(alpha=0.1))])

NB_clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = NB_clf.predict(X_test)

# Model Accuracy: how often is the classifier correct?
print("NaiveBayis Classifier Accuracy: ", metrics.accuracy_score(y_test, y_pred))
print("---------------------------------------")
# print("Predictions: ")
# print(NB_clf.predict(("LUCIA",)))
# print(NB_clf.predict(("BORJA",)))
# print(NB_clf.predict(("MATEO",)))
# print(NB_clf.predict(("VICTORIA",)))
# print(NB_clf.predict(("FEDERICO",)))
# print(NB_clf.predict(("DANIELA",)))
# print(NB_clf.predict(("ALEJANDRA",)))
# print(NB_clf.predict(("SAID",)))

######################################################################
# DecisionTreeClassifier

tree_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 1))),
                     ('tfidf', TfidfTransformer(norm='l2', sublinear_tf=True, use_idf=False)),
                     ('clf', DecisionTreeClassifier(criterion='entropy'))])

tree_clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = tree_clf.predict(X_test)

# Model Accuracy: how often is the classifier correct?
print("DecisionTreeClassifier Accuracy: ", metrics.accuracy_score(y_test, y_pred))
print("---------------------------------------")
# print("Predictions: ")
# print(tree_clf.predict(("LUCIA",)))
# print(tree_clf.predict(("BORJA",)))
# print(tree_clf.predict(("MATEO",)))
# print(tree_clf.predict(("VICTORIA",)))
# print(tree_clf.predict(("FEDERICO",)))
# print(tree_clf.predict(("DANIELA",)))
# print(tree_clf.predict(("ALEJANDRA",)))
# print(tree_clf.predict(("SAID",)))

################################################################################
# RandomForestClassifier

random_forest_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 1))),
                              ('tfidf', TfidfTransformer(norm='l2', sublinear_tf=True, use_idf=False)),
                              ('clf', RandomForestClassifier(criterion='entropy', random_state=7))])

# Train the model using the training sets
random_forest_clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = random_forest_clf.predict(X_test)

# Model Accuracy: how often is the classifier correct?
print("RandomForestClassifier Accuracy: ", metrics.accuracy_score(y_test, y_pred))
print("---------------------------------------")
# print("Predictions: ")
# print(random_forest_clf.predict(("LUCIA",)))
# print(random_forest_clf.predict(("BORJA",)))
# print(random_forest_clf.predict(("MATEO",)))
# print(random_forest_clf.predict(("VICTORIA",)))
# print(random_forest_clf.predict(("FEDERICO",)))
# print(random_forest_clf.predict(("DANIELA",)))
# print(random_forest_clf.predict(("ALEJANDRA",)))
# print(random_forest_clf.predict(("SAID",)))

#################################################################################

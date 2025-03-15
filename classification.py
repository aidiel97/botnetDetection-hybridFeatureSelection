import numpy as np

from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier

algorithmDict = {
  'decisionTree': DecisionTreeClassifier(criterion='entropy', max_depth=13),
  'logisticRegression' : LogisticRegression(),
  'naiveBayes': MultinomialNB(alpha=0.5, fit_prior=False),
  'adaboost': AdaBoostClassifier(n_estimators=600, learning_rate=1.0),
  'extraTree': ExtraTreesClassifier(n_estimators=400, criterion='entropy'),
  'xGBoost': GradientBoostingClassifier(),
  'randomForest': RandomForestClassifier(n_estimators=1000, criterion='entropy', max_features='log2', max_depth=13),
  'knn': KNeighborsClassifier(n_neighbors=13, metric='manhattan', weights='uniform'),
  'svc' : SVC(),
  'ann': MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
}

def menu():
    print("\n================================================================")
    print("===============| List of Available Classification Algoritm |=================")
    print("================================================================\n")
    keys_list = list(algorithmDict.keys())
    x=1
    for key in keys_list:
        print(str(x)+". "+key)
        x += 1

    selected_algo = int(input("Select the algoritm to use: "))
    return keys_list[selected_algo-1]

def train(X_train, y_train, algorithm):
    model = algorithmDict[algorithm]
    model.fit(X_train, y_train)

    return model
    
def test(model, X_test, y_test):
    labels = np.unique(y_test)
    y_pred = model.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=labels).ravel()
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("================================================================")
    print("===================| Classification Result |====================")
    print("================================================================\n")

    print("\nTrue Positive (TP):", tp)
    print("True Negative (TN):", tn)
    print("False Positive (FP):", fp)
    print("False Negative (FN):", fn)

    print("\nAccuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)

    return tp, tn, fp, fn, accuracy, precision, recall, f1
"""
Ce fichier implémente différentes fonctions pour utiliser des SVM
"""
# ----------------------------------------------------------------
"""
Cette fonction permet de sauvegarder les paramètres d'un svm
input : le nom du fichier dans lequel sauvegarder le modèle
output : aucun
"""


def save_svm(svc, filename="svm_parameters"):

    from sklearn.externals import joblib
    joblib.dump(svc, filename+'.pkl')
# ----------------------------------------------------------------


"""
Cette fonction permet de charger les paramètres d'un svm déjà sauvegardé
input : le nom du fichier depuis lequel charger le modèle
output : le svm
"""


def load_svm(filename="svm_parameters"):
    from sklearn.externals import joblib
    return joblib.load(filename+'.pkl')
# ------------------------------------------------------------------


"""
Cette fonction entraîne le svm à partir d'exemples
input : 
    -> training_cases : un array de taille [n_samples, n_features]
    -> labels : un array de taille [n_samples]
output : le svm
"""


def train_svm(training_cases, labels):
    from sklearn import svm
    my_svm = svm.SVC()
    my_svm.fit(training_cases, labels)
    return my_svm
# ------------------------------------------------------------------


"""
Cette fonction classifie les test_case
input : 
    -> test_cases : un array de taille [n_samples, n_features]
    -> labels : un array de taille [n_samples]
output : le vecteur des labels
"""


def svm_predict(my_svm, test_case):
    return my_svm.predict(test_case)

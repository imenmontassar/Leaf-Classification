import numpy as np
import pandas as pd

from sklearn.linear_model import Perceptron
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

class Classification_Lineaire(object):
    def __init__(self, x_train, y_train, cross_validation=False):
        #pour la validation croisée
        self.cross_validation = cross_validation
        
        #pour diviser nos données en respectant la proportion 
        strat = StratifiedShuffleSplit(n_splits=1, test_size=0.8, random_state=0).split(x_train, y_train)
        self.x_train, self.y_train, self.x_test, self.y_test = [[x_train.iloc[train, :], y_train[train], x_train.iloc[test, :], y_train[test]] for train, test in strat][0]

        #pour enregistrer et faire varier nos hyper paramêtres
        self.modele_perceptron = None
        self.penalty = 'l2'
        self.alpha = 0.0001
        self.max_iter = 400
        self.random_state = None  # essayer avec 1 aussi
        self.tol = 0.0001
        self.eta0 = 1

        #pour disposer des labels
        colone = []
        for etiquette in list(y_train):
            flag = True
            for j in range(len(colone)):
                if (colone[j] == etiquette):
                    flag = False
            if (flag):
                colone.append(etiquette)
        self.col = colone

    def entrainnement(self):
        if(self.cross_validation == True):
            self.cross_validation=False
            self.validation_croisee()
        # pour entrainner le modèle
        self.modele_perceptron = Perceptron(penalty=self.penalty,
                                             alpha=self.alpha,
                                             max_iter=self.max_iter, random_state=self.random_state,
                                             tol=self.tol, eta0=self.eta0).fit(self.x_train, self.y_train)

    def prediction(self):
        if(self.cross_validation == True):
            self.cross_validation = False
            self.validation_croisee()

        # pour prédire l'étiquettes d'une donnée de test
        return self.modele_perceptron.predict(self.x_test)


    #les hyper paramêtres sont recherché avec gridsearch de sklearn
    def validation_croisee(self):
        modele_perceptron = Perceptron(penalty=self.penalty,
                                       alpha=self.alpha,
                                       max_iter=self.max_iter, random_state=self.random_state,
                                       tol=self.tol, eta0=self.eta0).fit(self.x_train, self.y_train)

        parameters = {'penalty': ['l2'], 'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
                      'max_iter': [400, 500], 'tol': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
                      'eta0': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]}

        clf = GridSearchCV(modele_perceptron, parameters, cv=2)
        clf.fit(self.x_train, self.y_train)
        print(clf.best_params_)
        self.penalty = clf.best_params_["penalty"]
        self.alpha = clf.best_params_["alpha"]
        self.max_iter = clf.best_params_["max_iter"]
        self.tol = clf.best_params_["tol"]
        self.eta0 = clf.best_params_["eta0"]


    def resultats(self):
        #récupération de la prédiction
        predictions = self.prediction()
        #matrice de confusion
        matrice=pd.DataFrame(confusion_matrix(self.y_test, predictions, labels=self.col), index=self.col, columns=self.col)

        #récupération des données mal étiqueté
        classe_to_print=[]

        for k in range (matrice.shape[0]):
            if(matrice.iloc[k,k] != np.sum(matrice.iloc[k]) or matrice.iloc[k,k] != np.sum(matrice.iloc[:,k] )):
                classe_to_print.append(matrice.columns[k])

        #infos générales sur les classes non étiqueté
        sk_report = classification_report(
            digits=6,
            y_true=self.y_test,
            y_pred=predictions,
            labels=classe_to_print,
            zero_division=0
        )
        print(sk_report)

        print("precision = ", accuracy_score(self.y_test, predictions))

        print("matrice de confusion :")
        return matrice









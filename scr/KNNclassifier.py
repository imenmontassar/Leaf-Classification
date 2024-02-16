import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


class KNNClassifier(object):
    def __init__(self, x_train, y_train, cross_validation=False, x_test=None, test=False):
        #pour gérer la validation croisée
        self.cross_validation = cross_validation

        #pour s'adapter au format kaggle et au format de prédiction
        self.test = test

        if(test):
            self.x_train = x_train
            self.x_test = x_test
            self.y_train = y_train
        else:
            #sépare les données en respectant les proportions
            strat = StratifiedShuffleSplit(n_splits=1, test_size=0.8, random_state=0).split(x_train, y_train)
            self.x_train, self.y_train, self.x_test, self.y_test = [[x_train.iloc[train, :], y_train[train], x_train.iloc[test, :], y_train[test]] for train, test in strat][0]

        self.modele_knn = None
        #pour récupérer les labels
        colone = []
        for etiquette in list(y_train):
            flag = True
            for j in range(len(colone)):
                if (colone[j] == etiquette):
                    flag = False
            if (flag):
                colone.append(etiquette)
        self.col = colone
        #pour stoquer et modifier nos hyper-paramètres
        self.num_neighbors = 6
        self.weights = 'uniform'
        self.metric = 'euclidean'
        self.algo = 'auto'
        self.leaf_size = 5

    #pour entrainner le modèle (sklearn)
    def entrainnement(self):
        if(self.cross_validation == True):
            self.cross_validation=False
            self.validation_croisee()
        model_knn= KNeighborsClassifier(leaf_size=self.leaf_size,n_neighbors=self.num_neighbors,
                                        weights=self.weights, metric=self.metric, algorithm=self.algo)
        model_knn.fit(self.x_train, self.y_train)
        self.modele_knn = model_knn

    #deux modes de prédictions (vecteurs de labels ou matrice de probabilités selon les besoins)
    def prediction(self,flag=False):
        if (flag):
            return self.modele_knn.predict_proba(self.x_test)
        return self.modele_knn.predict(self.x_test)

    #avec gridsearch
    def validation_croisee(self):
        model_knn= KNeighborsClassifier(leaf_size=self.leaf_size,n_neighbors=self.num_neighbors,
                                        weights=self.weights, metric=self.metric, algorithm=self.algo)

        parameters = {'n_neighbors': [1, 2, 3, 4, 5, 6], 'weights': ['uniform', 'distance'],
                      'metric': ['euclidean', 'manhattan'], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], 'leaf_size': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
        clf = GridSearchCV(model_knn, parameters, cv=2)
        clf.fit(self.x_train, self.y_train)
        print(clf.best_params_)
        self.num_neighbors = clf.best_params_["n_neighbors"]
        self.weights = clf.best_params_["weights"]
        self.metric = clf.best_params_["metric"]
        self.algo = clf.best_params_["algorithm"]
        self.leaf_size = clf.best_params_["leaf_size"]


    def resultats(self):
        if(self.test):
            print("mettre les résultat sur le site")
            return None
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









import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

class Machine_Vecteurs_Support(object):

    # technique de classification.

    def __init__(self, x_train, y_train, cross_validation=False, x_test=None, test=False):
        #pour gérer la validation croisée
        self.cross_validation = cross_validation
        #pour s'adapter au format kaggle et au format de prédiction
        self.test=test
        if(test):
            self.x_train = x_train
            self.x_test = x_test
            self.y_train = y_train
        else:
            #sépare les données en respectant les proportions
            strat = StratifiedShuffleSplit(n_splits=1, test_size=0.8, random_state=0).split(x_train, y_train)
            self.x_train, self.y_train, self.x_test, self.y_test = [[x_train.iloc[train, :], y_train[train], x_train.iloc[test, :], y_train[test]] for train, test in strat][0]
        
        #pour récupérer les labels
        colone = []
        for etiquette in list(y_train):
            flag = True
            for j in range(len(colone)):
                if (colone[j] == etiquette):
                    flag = False
            if (flag):
                colone.append(etiquette)

        self.col=colone
        #pour stoquer et modifier nos hyper-paramètres
        self.modele_machine_vecteurs_support = None
        self.kernel='rbf'
        self.C= 1.0 # paramètre de régularisation, faire varier de 0.05 à 1.5
        self.gamma='scale' #scale ou auto

    #pour entrainner le modèle (sklearn)
    def entrainement(self):
        # pour entrainner le modèle
        if(self.cross_validation == True):
            self.cross_validation=False
            self.validation_croisee()

        modele_mvs = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            verbose=0,
            probability=True,
            random_state=None, )

        modele_mvs.fit(self.x_train, self.y_train)
        self.modele_machine_vecteurs_support = modele_mvs



    #deux modes de prédictions (vecteurs de labels ou matrice de probabilités selon les besoins)
    def prediction(self,flag=False):
        if(flag):
            return self.modele_machine_vecteurs_support.predict_proba(self.x_test)
        return self.modele_machine_vecteurs_support.predict(self.x_test)

    #avec gridsearch
    def validation_croisee(self):
        modele_mvs = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            verbose=0,
            probability=True,
            random_state=None, )
        parameters = {'C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1], 'gamma': ['auto', 'scale']}
        clf = GridSearchCV(modele_mvs, parameters, cv=2)
        clf.fit(self.x_train, self.y_train)
        print(clf.best_params_)
        self.C = clf.best_params_["C"]
        self.gamma = clf.best_params_["gamma"]

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









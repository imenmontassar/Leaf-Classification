import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

class Boosting_Modele(object):

    def __init__(self, x_train, y_train, cross_validation=False):
        #pour la validation croisée
        self.cross_validation = cross_validation
        #pour diviser nos données en respectant la proportion 
        strat = StratifiedShuffleSplit(n_splits=1, test_size=0.8, random_state=0).split(x_train, y_train)
        self.x_train, self.y_train, self.x_test, self.y_test = [[x_train.iloc[train, :], y_train[train], x_train.iloc[test, :], y_train[test]] for train, test in strat][0]

        #pour disposer des labels
        colone = []
        for etiquette in list(y_train):
            flag = True
            for j in range(len(colone)):
                if (colone[j] == etiquette):
                    flag = False
            if (flag):
                colone.append(etiquette)
        self.col=colone

        #pour enregister le modèle
        self.modele_combinaison_modeles = None
        self.base_estimator=DecisionTreeClassifier()

        #pour enregistrer et faire varier nos hyper paramêtres
        self.n_estimators=50
        self.learning_rate= 1.0 # paramètre d'apprenrtissage, faire varier de 0.25 à 1.5
        self.random_state = None # essayer avec 1 aussi

    #entrainne le modèle avec les fonctions sklearn
    def entrainement(self):
        # ici in regarde si une validation croisée est demandé
        if(self.cross_validation == True):
            self.cross_validation=False
            self.validation_croisee()

        modele_cm = AdaBoostClassifier(
            base_estimator=self.base_estimator,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            algorithm='SAMME',
            random_state=self.random_state, )

        modele_cm.fit(self.x_train, self.y_train)
        self.modele_combinaison_modeles = modele_cm

    #pour gérer la prédiction
    def prediction(self):
        return self.modele_combinaison_modeles.predict(self.x_test)

    #les hyper paramêtres sont recherché avec gridsearch de sklearn
    def validation_croisee(self):
        modele_cm = AdaBoostClassifier(
            base_estimator=self.base_estimator,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            algorithm='SAMME',
            random_state=self.random_state, )
        parameters = {'n_estimators': np.arange(30, 150, 20), 'learning_rate': np.linspace(0.25, 1.5, 6),
                      'random_state': [None, 1]}

        clf = GridSearchCV(modele_cm, parameters, cv=2)
        clf.fit(self.x_train, self.y_train)
        print(clf.best_params_)
        self.n_estimators = clf.best_params_["n_estimators"]
        self.learning_rate = clf.best_params_["learning_rate"]
        self.random_state = clf.best_params_["random_state"]


    #pour afficher les résultats dans le notebook
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


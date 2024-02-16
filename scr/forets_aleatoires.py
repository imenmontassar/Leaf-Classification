import numpy as np
import pandas as pd


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


class Forets_Aleatoires(object):


    def __init__(self, x_train, y_train, cross_validation=False, x_test=None, test=False):
        #pour gérer la validation croisée
        self.cross_validation = cross_validation
        #pour s'adapter au format kaggle et au format de prédiction
        self.test = test

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

        if(test):
            self.x_train = x_train
            self.x_test = x_test
            self.y_train = y_train
        else:
            #sépare les données en respectant les proportions
            strat = StratifiedShuffleSplit(n_splits=1, test_size=0.8, random_state=0).split(x_train, y_train)
            self.x_train, self.y_train, self.x_test, self.y_test = [[x_train.iloc[train, :], y_train[train], x_train.iloc[test, :], y_train[test]] for train, test in strat][0]

        #pour stoquer et modifier nos hyper-paramètres
        self.modele_random_forest = None
        self.nmb_arbre = 200 # faire varier de 100 à 500
        self.criterion = 'gini' # gini et entropy
        self.min_samples_split = 2 # interessant de faire varier des fois moins séparer permet de mieux classifier

    #pour entrainner le modèle (sklearn)
    def entrainement(self):

        if(self.cross_validation == True):
            self.cross_validation == False
            self.validation_croisee()

        modele_rf = RandomForestClassifier(
            n_estimators=self.nmb_arbre,
            criterion=self.criterion,
            max_depth=None,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features='auto',
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=True,
            oob_score=False,
            n_jobs=None,
            random_state=None,
            verbose=0,
            warm_start=False,
            class_weight=None,
            ccp_alpha=0.0,
            max_samples=None, )

        modele_rf.fit(self.x_train, self.y_train)
        self.modele_random_forest = modele_rf

    #deux modes de prédictions (vecteurs de labels ou matrice de probabilités selon les besoins)
    def prediction(self,flag=False):
        if(flag):
            return self.modele_random_forest.predict_proba(self.x_test)
        return self.modele_random_forest.predict(self.x_test)

    #avec gridsearch
    def validation_croisee(self):
        modele_rf = RandomForestClassifier(
            n_estimators=self.nmb_arbre,
            criterion=self.criterion,
            max_depth=None,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features='auto',
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=True,
            oob_score=False,
            n_jobs=None,
            random_state=None,
            verbose=0,
            warm_start=False,
            class_weight=None,
            ccp_alpha=0.0,
            max_samples=None, )

        parameters = {'n_estimators': np.arange(100, 702, 100), 'criterion': ['gini', 'entropy'],
                      'min_samples_split': [2, 3, 4]}
        clf = GridSearchCV(modele_rf, parameters, cv=2)
        clf.fit(self.x_train, self.y_train)
        print(clf.best_params_)
        self.nmb_arbre = clf.best_params_["n_estimators"]
        self.criterion = clf.best_params_["criterion"]
        self.min_samples_split = clf.best_params_["min_samples_split"]

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









import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

class Reseaux_Neuronnes(object):

    # technique de classification.

    def __init__(self, x_train, y_train, cross_validation=False, x_test=None, test=False):
        #pour gérer la validation croisée
        self.cross_validation = cross_validation
        #pour récupérer les labels
        self.colone = None
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


        self.modele_reseau_neuronnes = None
        #pour stoquer et modifier nos hyper-paramètres
        self.hidden_layer_sizes = (100,)
        self.activation = 'relu' # les autres sont moins bien cf cours
        self.alpha = 0.0001
        self.max_iter = 200
        self.random_state = None # essayer avec 1 aussi
        self.tol = 0.0001

    #pour entrainner le modèle (sklearn)
    def entrainement(self):

        if(self.cross_validation == True):
            self.cross_validation = False
            self.validation_croisee()

        self.modele_reseau_neuronnes = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes,
                                                     activation=self.activation, alpha=self.alpha,
                                                     max_iter=self.max_iter, random_state=self.random_state,
                                                     tol=self.tol).fit(self.x_train, self.y_train)

    #deux modes de prédictions (vecteurs de labels ou matrice de probabilités selon les besoins)
    def prediction(self, flag=False):
        if(flag):
            return self.modele_reseau_neuronnes.predict_proba(self.x_test)

        predictions = self.modele_reseau_neuronnes.predict_proba(self.x_test)
        self.colone=self.modele_reseau_neuronnes.classes_
        etiquettes_predits = []
        for i in range (predictions.shape[0]):
            tmp = max(predictions[i])
            etiquettes_predits.append(self.colone[list(predictions[i]).index(tmp)])
        return np.array(etiquettes_predits)

    #avec gridsearch
    def validation_croisee(self):

        modele_rf =  MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes,
                                   activation=self.activation, alpha=self.alpha,
                                   max_iter=self.max_iter, random_state=self.random_state,
                                   tol=self.tol).fit(self.x_train, self.y_train)

        parameters = {'hidden_layer_sizes': [(100,), (350,), (200,), (150,)], 'max_iter': [400, 600, 800],
                      'random_state': [None, 1]}

        clf = GridSearchCV(modele_rf, parameters, cv=2)
        clf.fit(self.x_train, self.y_train)
        print(clf.best_params_)
        self.hidden_layer_sizes = clf.best_params_["hidden_layer_sizes"]
        self.max_iter = clf.best_params_["max_iter"]
        self.random_state = clf.best_params_["random_state"]


    def resultats(self):
        if(self.test):
            print("mettre les résultat sur le site")
            return None

        predictions = self.prediction()

        matrice=pd.DataFrame(confusion_matrix(self.y_test, predictions, labels=self.colone), index=self.colone, columns=self.colone)

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

        print("precision = ", self.modele_reseau_neuronnes.score(self.x_test, self.y_test))

        print("matrice de confusion :")
        return matrice
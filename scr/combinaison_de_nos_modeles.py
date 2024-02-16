import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report

from machine_vecteurs_support import Machine_Vecteurs_Support
from reseaux_neuronnes import Reseaux_Neuronnes
from forets_aleatoires import Forets_Aleatoires
from KNNclassifier import KNNClassifier
from sklearn.metrics import classification_report


class Combinaisons(object):


    def __init__(self, x_train, y_train, cross_validation=False, x_test=None, test=False):

        #pour la cross validation
        self.cross_validation = cross_validation

        # récupération des colones
        colone = []
        for etiquette in list(y_train):
            flag = True
            for j in range(len(colone)):
                if (colone[j] == etiquette):
                    flag = False
            if (flag):
                colone.append(etiquette)
        self.col=colone

        #séparation des données en deux ensembles pour les tests
        if(test):
            self.x_train = x_train
            self.x_test = x_test
            self.y_train = y_train
        else:
            perm = np.random.permutation(len(x_train))
            X = x_train.to_numpy()[perm]
            t = y_train.to_numpy()[perm]

            self.x_train = X[0:int((len(X) * 80) / 100)]
            self.x_test = X[int((len(X) * 80) / 100):]
            self.y_train = t[0:int((len(X) * 80) / 100)]
            self.y_test = t[(int((len(X) * 80) / 100)):]

        #pour stoquer nos modèles, nous leur donnons les meilleurs hyper paramêtre de nos cross validation
        self.modele_foret_aleatoire = Forets_Aleatoires(self.x_train, self.y_train, False, self.x_test, True)
        self.modele_foret_aleatoire.nmb_arbre=600
        self.modele_foret_aleatoire.criterion="entropy"
        self.modele_foret_aleatoire.min_samples_split=4
        self.pond_foret_aleatoire = 0.89
        self.modele_machine_vecteurs_support = Machine_Vecteurs_Support(self.x_train, self.y_train, False, self.x_test, True)
        self.pond_machine_vecteurs_support = 0.69
        self.modele_machine_vecteurs_support.C = 1e-05
        self.modele_machine_vecteurs_support.gamma = 'auto'
        self.modele_reseaux_neuronnes = Reseaux_Neuronnes(self.x_train, self.y_train, False, self.x_test, True)
        self.modele_reseaux_neuronnes.hidden_layer_sizes=(200,)
        self.modele_reseaux_neuronnes.max_iter=800
        self.modele_reseaux_neuronnes.random_state=None
        self.pond_reseaux_neuronnes = 0.81
        self.modele_knn = KNNClassifier(self.x_train, self.y_train, False, self.x_test, True)
        self.modele_knn.algo="auto"
        self.modele_knn.leaf_size=1
        self.modele_knn.metric="manhattan"
        self.modele_knn.num_neighbors=1
        self.modele_knn.weights="uniform"
        self.pond_knn = 0.86

    #on entrainne tous nos modèles
    def entrainement(self):
        self.modele_reseaux_neuronnes.entrainement()
        self.modele_foret_aleatoire.entrainement()
        self.modele_machine_vecteurs_support.entrainement()
        self.modele_knn.entrainnement()

    #pour la prédiction nous faisons une moyenne pondéré des résultats de chaque classe selon leur prédiction
    def prediction(self,flag=False):

        pred_reseaux_neuronnes = self.modele_reseaux_neuronnes.prediction(True)
        pred_foret_aleatoire = self.modele_foret_aleatoire.prediction(True)
        pred_machine_vecteurs_support = self.modele_machine_vecteurs_support.prediction(True)
        pred_knn = self.modele_knn.prediction(True)
        if(flag):
            pred=pred_reseaux_neuronnes*self.pond_reseaux_neuronnes+pred_foret_aleatoire*self.pond_foret_aleatoire+pred_machine_vecteurs_support*self.pond_machine_vecteurs_support+pred_knn*self.pond_knn/(self.pond_knn+self.pond_machine_vecteurs_support+self.pond_foret_aleatoire+self.pond_reseaux_neuronnes)
            return pred

        predictions = pred_reseaux_neuronnes*self.pond_reseaux_neuronnes+pred_foret_aleatoire*self.pond_foret_aleatoire+pred_machine_vecteurs_support*self.pond_machine_vecteurs_support+pred_knn*self.pond_knn
        self.colone=self.modele_reseaux_neuronnes.modele_reseau_neuronnes.classes_
        etiquettes_predits = []
        for i in range (predictions.shape[0]):
            tmp = max(predictions[i])
            print(tmp)
            etiquettes_predits.append(self.colone[list(predictions[i]).index(tmp)])
        return etiquettes_predits


    # la validation croisée est faite au préalable
    def validation_croisee(self):
        print("pas ici se serait trop long")


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






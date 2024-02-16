import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
from scipy import stats
import pandas as pd
class Pretraitement(object):

    # effectue le prétraitement sur nos données, ne retourne jamais rien, modifie self.x_train.

    def __init__(self, x_train):
        if 'species' in x_train.columns:
            x_train = x_train.drop(['species'], axis=1)
        if 'id' in x_train.columns:
            x_train = x_train.drop(['id'], axis=1)
        self.x_train = x_train # nos données d'entrainement sans étiquette sur lesquelles on veut appliquer nos prétraitements

    def normaliser(self):
        # normalise les données entre 0 et 1
        X = self.x_train # à supprimer
        #Utilisation de la méthode min Max pour normaliser les données
        scaler=preprocessing.MinMaxScaler()
        #Normalisation des valeurs
        d=scaler.fit_transform(X)
        #Retransformation en dataframe
        final_data=pd.DataFrame(d,columns=names)
        
        return(final_data)



    def selection_avec_PCA(self,ratio:float):
        # applique la méthode PCA sur nos données et conserve les nmb_col colonnes les plus discriminantes
        X = self.x_train.to_numpy()
        taille = min(X.shape[1], X.shape[0])
        pca = PCA(n_components=taille)
        pca.fit(X)
        nmb_col = 20
        sum = 0
        while sum<ratio:
            sum = np.sum(pca.explained_variance_ratio_[:nmb_col])
            nmb_col += 1

        print("pca variance ratio: ", sum)
        print("nmb de colonne : ", nmb_col)
        pca_finale = PCA(n_components=nmb_col)
        pca_finale.fit(X)
        self.x_train = pd.DataFrame(pca_finale.transform(X))


    def supprimer_donnees_aberrantes(self,seuil:float,):
        # supprime les données aberante c'est à dire celle tel que: Z-score>seuil
        #Supprime les données dont 10% des attributs est considéré comme valeur aberrante

        #1 Calcul Z score, on le stocke 
        z_scores=stats.zscore(self.x_train)
        #2 valeur absolue de ces valeurs, puisque un écart même négatif représente un outlier
        abs_z_scores = np.abs(z_scores)
        #3 Recupération des indices des outliers
        #dépend du seuil, ici on prend les données dont 10% des attributs sont considérés comme outlier
        Outliers_list=[]
        for i in range (abs_z_scores.shape[0]):
            #pour chaque attribut (192)
            compt=0
            for j in range (abs_z_scores.shape[1]):
                if (abs_z_scores.iloc[i,j]>seuil) :
                    compt=compt+1
            if (compt>0.1*abs_z_scores.shape[1]):
                print(i)
                #print(abs_z_scores.loc[i])
                Outliers_list.append(i)
        #3 On ne garde que les données qui ne sont pas des outliers
        Filt=self.x_train[~self.x_train.index.isin(Outliers_list)] 
        return(Filt)




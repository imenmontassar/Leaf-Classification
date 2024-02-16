#Librairies
import pandas as pd
import skimage.io as io
import numpy as np
from numpy import savetxt
from skimage.transform import resize

#à l'aide de scikit-image nous effectuons nos prétraitement sur nos images dans ce fichier

#pour récuperer nos images en fonctions des ids de nos csv
def load_images(root, list_id):
    list_images = []
    for i in list_id:
        print(i)
        image = io.imread(root + str(i) + '.jpg')
        size = max(image.shape)
        # 0-pading
        image = np.pad(image, ((size-image.shape[0], 0), (size-image.shape[1], 0)), 'constant')
        # redimentionnement
        image_resized = resize(image, (50, 50), preserve_range=True, anti_aliasing=True)
        list_images.append(image_resized)
    io.imsave("after.jpg", list_images[1])
    io.imsave("after2.jpg", list_images[7])
    return list_images

# appel des csv, éxécutions et sauvegarde des tableaux de nos pixels
df_train = pd.read_csv('../leaf-classification/train.csv')
df_test = pd.read_csv('../leaf-classification/test.csv')
print("the len of the train set is:",len(df_train))
print("the len of the test set is:",len(df_test))
root = "../images/"

train_ID = df_train['id']
train_Y = df_train['species']
test_ID = df_test['id']

liste_image_test = load_images(root, test_ID)
liste_image_test = np.array(liste_image_test)
liste_image_test = liste_image_test.reshape(len(liste_image_test), -1)
savetxt('../leaf-classification/test_image.csv',liste_image_test, delimiter=',')

liste_image_train = load_images(root, train_ID)
liste_image_train = np.array(liste_image_train)
liste_image_train = liste_image_train.reshape(len(liste_image_train), -1)
savetxt('../leaf-classification/train_image.csv',liste_image_train, delimiter=',')

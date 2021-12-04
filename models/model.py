# Chargement des bibliothèques
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


# constant
DATA_TRANSFORM = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                  'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
ENC = LabelEncoder()

class BasePredector:

    def __init__(self) -> None:

        # initial data
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.data = pd.read_csv('Telco_Customer_Churn.csv').drop(['customerID'],axis=1)
        self.data["Churn"] = ENC.fit_transform(self.data["Churn"])
        self._convert_object_to_num()
        self._prepare_data()
        self.split_data() 

    def _convert_object_to_num(self) -> None:
        print(f"start converting type object to num")
        for column in DATA_TRANSFORM:
            self.data[column] = ENC.fit_transform(self.data[column])
        print(f"convert total chager to float !")
        self.data["TotalCharges"] = self.data["TotalCharges"].apply(lambda x: x.strip()).replace('', np.nan).astype(float)
        print(f"remove nan values !")
        self.data = self.data.dropna(axis=0)   

    def _prepare_data(self) -> None:
        print(f"Prepare X and y start ...")
        self.X = self.data.drop(["Churn"],axis=1)
        print(f"Shape X : {self.X.shape}")
        self.y = self.data.Churn

    def split_data(self):
        self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=0)



class KNNPredector(BasePredector):

    def __init__(self) -> None:
        super().__init__()
        # self.grid_param = {'n_neighbors':np.arange(1,25),'metric':['euclidean','manhattan','minkowski']}
        # self.grid: GridSearchCV = GridSearchCV(KNeighborsClassifier(),self.grid_param,cv=5)
        # self.grid.fit(self.X_train,self.y_train)
        # print(f"Le meilleur score : {self.grid.best_score_}")
        # print(f"Les meilleurs valeurs des hyper parametres : {self.grid.best_params_}") 
        # self.final_model_knn: KNeighborsClassifier = self.grid.best_estimator_
        self.final_model_knn = KNeighborsClassifier(metric='euclidean', n_neighbors=11)
        self.final_model_knn.fit(X=self.X_train,y=self.y_train)


    def __str__(self) -> str:
        return "knn model"


    def predict(self,X: pd.Series):
        y_pred = self.final_model_knn.predict([X])
        return y_pred

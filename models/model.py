# Chargement des bibliothèques
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier



# constant
DATA_TRANSFORM = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                  'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
ENC = LabelEncoder()

class BasePredictor:

    def __init__(self):

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

    def _convert_object_to_num(self):
        print(f"start converting type object to num")
        for column in DATA_TRANSFORM:
            self.data[column] = ENC.fit_transform(self.data[column])
        print(f"convert total chager to float !")
        self.data["TotalCharges"] = self.data["TotalCharges"].apply(lambda x: x.strip()).replace('', np.nan).astype(float)
        print(f"remove nan values !")
        self.data = self.data.dropna(axis=0)   

    def _prepare_data(self):
        print(f"Prepare X and y start ...")
        self.X = self.data.drop(["Churn"],axis=1)
        print(f"Shape X : {self.X.shape}")
        self.y = self.data.Churn

    def split_data(self):
        self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=0)



class KNNPredictor(BasePredictor):

    def __init__(self):
        super().__init__()
        # self.grid_param = {'n_neighbors':np.arange(1,25),'metric':['euclidean','manhattan','minkowski']}
        # self.grid: GridSearchCV = GridSearchCV(KNeighborsClassifier(),self.grid_param,cv=5)
        # self.grid.fit(self.X_train,self.y_train)
        # print(f"Le meilleur score : {self.grid.best_score_}")
        # print(f"Les meilleurs valeurs des hyper parametres : {self.grid.best_params_}") 
        # self.final_model_knn: KNeighborsClassifier = self.grid.best_estimator_
        self.final_model_knn = KNeighborsClassifier(metric='euclidean', n_neighbors=11) # best params
        self.final_model_knn.fit(X=self.X_train,y=self.y_train)


    def __str__(self):
        return "knn model"


    def predict(self,X: pd.Series):
        y_pred = self.final_model_knn.predict([X])
        return y_pred

class DecisionTreePredictor(BasePredictor):
    
    def __init__(self):
        super().__init__()
        self.final_model_dt = DecisionTreeClassifier(criterion='entropy', max_depth=6, random_state=0) # best params
        self.final_model_dt.fit(X=self.X_train,y=self.y_train)

    def __str__(self):
        return "decision tree model"

    def predict(self,X: pd.Series):
        y_pred = self.final_model_dt.predict([X])
        return y_pred

class LogisticRegressionPredictor(BasePredictor):
    
    def __init__(self):
        super().__init__()
        self.final_model_lr = LogisticRegression(C=0.21052631578947345, penalty='l1', random_state=0,solver='liblinear') # best params
        self.final_model_lr.fit(X=self.X_train,y=self.y_train)

    def __str__(self):
        return "logistic regression model"

    def predict(self,X: pd.Series):
        y_pred = self.final_model_lr.predict([X])
        return y_pred

class GaussianNBPredictor(BasePredictor):
    
    def __init__(self):
        super().__init__()
        self.final_model_nb = GaussianNB(var_smoothing=0.0001232846739442066) # best params
        self.final_model_nb.fit(X=self.X_train,y=self.y_train)

    def __str__(self):
        return "GaussianNB model"

    def predict(self,X: pd.Series):
        y_pred = self.final_model_nd.predict([X])
        return y_pred

class SVMPredictor(BasePredictor):
    
    def __init__(self):
        super().__init__()
        self.final_model_svm = SVC(C=10, gamma=0.0001, probability=True) # best params
        self.final_model_svm.fit(X=self.X_train,y=self.y_train)

    def __str__(self):
        return "GaussianNB model"

    def predict(self,X: pd.Series):
        y_pred = self.final_model_svm.predict([X])
        return y_pred

class RandomForestPredictor(BasePredictor):
    
    def __init__(self):
        super().__init__()
        self.final_model_rf = RandomForestClassifier(criterion='entropy', max_depth=6, random_state=0) # best params
        self.final_model_rf.fit(X=self.X_train,y=self.y_train)

    def __str__(self):
        return "RandomForestClassifier model"

    def predict(self,X: pd.Series):
        y_pred = self.final_model_rf.predict([X])
        return y_pred

class XGBPredictor(BasePredictor):
    
    def __init__(self):
        super().__init__()
        self.final_model_xgb = XGBClassifier(colsample_bytree=0.6, gamma=5, max_depth=4, min_child_weight=10, subsample=1.0) # best params
        self.final_model_xgb.fit(X=self.X_train,y=self.y_train)

    def __str__(self):
        return "XGBClassifier model"

    def predict(self,X: pd.Series):
        y_pred = self.final_model_xgb.predict([X])
        return y_pred

class FinalModelPredictor:
    def __init__(self):
        self.final_model_xgb = XGBPredictor()
        self.final_model_rf = RandomForestPredictor()
        self.final_model_nb = GaussianNBPredictor()
        self.final_model_lr = LogisticRegressionPredictor()
        self.final_model_svm = SVMPredictor()
        self.final_model_dt = DecisionTreePredictor()
        self.final_model_knn = KNNPredictor()

    def predict(self,X: pd.Series):
        print(f"Start predict !")
        result: dict = {
            "knn" : self.final_model_knn.predict(X=X)[0],
            "xgb" : self.final_model_xgb.predict(X=X)[0],
            "rf" : self.final_model_rf.predict(X=X)[0],
            "nb" : self.final_model_nb.predict(X=X)[0],
            "lr" : self.final_model_lr.predict(X=X)[0],
            "svm" : self.final_model_svm.predict(X=X)[0],
            "dt" : self.final_model_dt.predict(X=X)[0],
        }
        return result 
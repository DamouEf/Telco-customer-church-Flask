from flask import Flask
from flask import request
from models.model import KNNPredector
import pandas  as pd
import json
app = Flask(__name__)

# constants
FINAL_MODEL = None
GENDER: dict = {
    "female": 0,
    "male": 1
}
BOOL: dict = {
    "yes": 1,
    "no": 0,
}
MULTIPLE_LINES: dict = {
    'yes': 2,
    'no': 0,
    'no phone service': 1
}
INTERNET_SERVICE: dict = {
    "dsl": 0,
    "fibre optic": 1,
    "no": 2
}
INTERNET: dict = {
    'yes': 2,
    'no': 0,
    'no internet service': 1
}
CONTRACT: dict = {
    "month-to-month": 0,
    "one year": 1,
    "two year": 2
}
PAYEMENT_METHOD: dict = {
    "bank transfer (automatic)": 0,
    "credit card (automatic)":1,
    "electronic check" : 2,
    "mailed check" : 3,
}

@app.route('/predict', methods=['POST'])
def predict():
    """
    This API takes as input :
        payload = {
            "gender": Female/Male,
            "senior_citizen": yes,/no,
            "partner": Yes/No,
            "dependents": Yes/No,
            "tenure": number,
            "phone_service": Yes/No,
            "multiple_lines": Yes/No/No phone service,
            "internet_service": dsl/fibre optic/no ,
            "online_security": Yes/No/No internet service,
            "online_backup":  Yes/No/No internet service,
            "device_protection": Yes/No/No internet service,
            "tech_support": Yes/No/No internet service,
            "streaming_tv": Yes/No/No internet service,
            "streaming_movies": Yes/No/No internet service,
            "contract": month-to-month/one year/two year,
            "paperless_billing": yes/no,
            "payment_method": ,
            "monthly_charges": float,
            "totales_charges": ,
        }
    """

    if request.data:
        payload: dict = json.loads(request.data)
        x_to_predict = pd.Series(
            [
                GENDER[payload.get('gender')],
                BOOL[payload.get('senior_citizen')],
                BOOL[payload.get('partner')],
                BOOL[payload.get('dependents')],
                payload.get('tenure'),
                BOOL[payload.get('phone_service')],
                MULTIPLE_LINES[payload.get('multiple_lines')],
                INTERNET_SERVICE[payload.get('internet_service')],
                INTERNET[payload.get('online_security')],
                INTERNET[payload.get('online_backup')],
                INTERNET[payload.get('device_protection')],
                INTERNET[payload.get('tech_support')],
                INTERNET[payload.get('streaming_tv')],
                INTERNET[payload.get('streaming_movies')],
                CONTRACT[payload.get('contract')],
                BOOL[payload.get('paperless_billing')],
                PAYEMENT_METHOD[payload.get('payment_method')],
                payload.get('monthly_charges'),
                payload.get('totales_charges'),
                ],
            index=[
                'gender',
                'SeniorCitizen',
                'Partner',
                'Dependents',
                'tenure',
                'PhoneService',
                'MultipleLines',
                'InternetService',
                'OnlineSecurity',
                'OnlineBackup',
                'DeviceProtection',
                'TechSupport',
                'StreamingTV',
                'StreamingMovies',
                'Contract',
                'PaperlessBilling',
                'PaymentMethod',
                'MonthlyCharges',
                "TotalCharges"])

    # TODO use FINALE_MODEL to predect result
    result = FINAL_MODEL.predict(x_to_predict)

    message: str = f"x to predect :\n{x_to_predict.__dict__}\nresult:\n{result}"

    return message
    # TODO get data from request
  
  
if __name__ == "__main__":

    print(f"Application start !")
    print(f"start entrain model ...")

    FINAL_MODEL = KNNPredector()
    app.run(host ='0.0.0.0', port=5000, debug=True) 


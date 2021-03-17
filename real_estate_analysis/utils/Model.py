import joblib


class ModelObject:
    """
    Object to hold a model, data transformation pipeline, and its associated scores for future comparison
    """

    def __init__(self, model, scores):
        self.model = model
        self.scores = scores
        self.imputer = self.load_imputer()
        self.scaler = self.load_scaler()

    def __repr__(self):
        return f"ModelObject('{self.model})"

    def __str__(self):
        return f'{self.model} object with associated scores'

    def get_model(self):
        return self.model

    def get_scores(self):
        return self.scores

    def get_imputer(self):
        return self.imputer

    def get_scaler(self):
        return self.imputer

    @staticmethod
    def load_imputer():
        with open('../../data/models/sold/imputer.joblib', 'rb') as file:
            imp = joblib.load(file)
        return imp

    @staticmethod
    def load_scaler():
        with open('../../data/models/sold/scaler.joblib', 'rb') as file:
            scaler = joblib.load(file)
        return scaler

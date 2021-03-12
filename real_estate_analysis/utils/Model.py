class ModelObject:
    """
    Object to hold a model and its associated scores for future comparison
    """

    def __init__(self, model, scores):
        self.model = model
        self.scores = scores

    def __repr__(self):
        return f"ModelObject('{self.model})"

    def __str__(self):
        return f'{self.model} object with associated scores'

    def get_model(self):
        return self.model

    def get_scores(self):
        return self.scores

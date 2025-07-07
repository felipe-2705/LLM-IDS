from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
import xgboost as xgb


class ALGORITHM:
    """
    Dictionary of algorithms to be used in the model.
    Each key is the name of the algorithm and the value is a lambda function that returns an instance of the classifier.
    """
    def __init__(self):
        self.algorithms = {
            'knn': lambda: KNeighborsClassifier(),
            'dt': lambda: DecisionTreeClassifier(random_state=42),
            'nb': lambda: GaussianNB(var_smoothing=1e-9),
            'svm': lambda: SVC(),
            'rf': lambda: RandomForestClassifier(random_state=42),
            'xgboost': lambda: xgb.XGBClassifier(eval_metric='mlogloss', random_state=42),
            'linear_svc': lambda: LinearSVC(max_iter=1000, random_state=42),
            'sgd': lambda: SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
        }
'''knn.py

K-nearest neighbors
'''
import pandas as pd
import logging 
import numpy as np
from experiment import Experiment
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
logger = logging.getLogger(__name__)


from sklearn.metrics import make_scorer, accuracy_score
from sklearn.utils import compute_sample_weight

def balanced_accuracy(truth,pred):
    wts = compute_sample_weight('balanced',truth)
    return accuracy_score(truth,pred,sample_weight=wts)
scorer = make_scorer(balanced_accuracy) 

class KNN(Experiment):

  def __init__(self, attributes, classifications, **kwargs):
      ''' Construct the object
      '''
      super().__init__(attributes, classifications)

  def _pipeline(self):
      '''Get the model pipeline (which classifier are you using
      '''
      return Pipeline([('scale', StandardScaler()), ('predict', KNeighborsClassifier())])

  def _params(self):
      ''' Get the model params of the expierment
      '''
      return {
        'predict__metric':['manhattan','euclidean','chebyshev'],
        'predict__n_neighbors': np.arange(1, 30, 3),
        'predict__weights': ['uniform','distance']
      }

  def run(self):
      ''' Run the experiment
      '''
      super().run()
      #print(cv.best_estimator_)
      print(cv.cv_results_)
      #print(len(cv.cv_results_['params']))
      #print(cv.cv_results_['split1_test_score'].shape)
      #print(cv.score(x_test, y_test))
      #curve = ms.learning_curve(cv.best_estimator_,trgX,trgY,cv=5,train_sizes=[50,100]+[int(N*x/10) for x in range(1,8)],verbose=10,scoring=scorer)
      #curve_train_scores = pd.DataFrame(index = curve[0],data = curve[1])
      #curve_test_scores  = pd.DataFrame(index = curve[0],data = curve[2])
      #curve_train_scores.to_csv('./output/{}_{}_LC_train.csv'.format(clf_type,dataset))
      #curve_test_scores.to_csv('./output/{}_{}_LC_test.csv'.format(clf_type,dataset))

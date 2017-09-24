'''boosting.py


Boosting with decision trees
'''
import logging 
import numpy as np
from dt import PrunedDecisionTreeClassifier
from experiment import Experiment
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.utils import compute_sample_weight
from sklearn.model_selection import cross_val_score
import warnings
import matplotlib.pyplot as plt
import pandas as pd

def balanced_accuracy(truth,pred):
    wts = compute_sample_weight('balanced', truth)
    return accuracy_score(truth, pred, sample_weight=wts)
scorer = make_scorer(balanced_accuracy) 
logger = logging.getLogger(__name__)

class Boosting(Experiment):

  def __init__(self, attributes, classifications, dataset, **kwargs):
      ''' Construct the object
      '''
      if dataset == 'wine':
          basic_tree = PrunedDecisionTreeClassifier(criterion='entropy', class_weight='balanced', random_state=10)
      else:
          basic_tree = PrunedDecisionTreeClassifier(criterion='gini', class_weight='balanced', random_state=10)

      learning_curve_train_sizes = np.arange(0.01, 1.0, 0.025)
      pipeline = Pipeline([('scale', StandardScaler()), 
        ('predict', AdaBoostClassifier(algorithm='SAMME', base_estimator=basic_tree, random_state=10))])
      params = {
        'predict__n_estimators': np.power(2, np.arange(1, 8)),
        'predict__base_estimator__alpha': np.append(np.arange(0.99, 1.03, 0.001), 0)
      }
      super().__init__(attributes, classifications, dataset, 'boosting', pipeline, params, 
        learning_curve_train_sizes, True, verbose=1, iteration_curve=False)

  def run(self):
      cv = super().run()
      logger.info('Running estimator check')
      n_estimators = np.arange(1, 24, 2)
      train_iter = []
      estimator_iter = []
      final_df = []
      with warnings.catch_warnings():
          warnings.simplefilter("ignore")
          best_estimator = cv.best_estimator_
          x_train, x_test, y_train, y_test = self._split_train_test()
          for i, n_estimator in enumerate(n_estimators):
              best_estimator.set_params(**{ 'predict__n_estimators': n_estimator })
              best_estimator.fit(x_train, y_train)
              train_iter.append(np.mean(cross_val_score(best_estimator, x_train, y_train, scoring=scorer, cv=self._cv)))
              estimator_iter.append(np.mean(cross_val_score(best_estimator, x_test, y_test, scoring=scorer, cv=self._cv)))
              final_df.append([n_estimator, train_iter[i], estimator_iter[i]])
          plt.figure(5)
          plt.plot(n_estimators, train_iter, 
            marker='o', color='blue', label='Train Score')
          plt.plot(n_estimators, estimator_iter, 
            marker='o', color='green', label='Test Score')
          plt.legend()
          plt.grid(linestyle='dotted')
          plt.xlabel('Estimators')
          plt.ylabel('Accuracy')
          csv_str = '{}/{}'.format(self._dataset, self._algorithm)
          plt.savefig('./results/{}/estimator-curve.png'.format(csv_str))
          iter_csv = pd.DataFrame(data = final_df, columns=['Estimators', 'Train Accuracy', 'Test Accuracy'])
          iter_csv.to_csv('./results/{}/estimator.csv'.format(csv_str), index=False)


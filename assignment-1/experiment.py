'''experiment.py

Share all of the experiment items here
'''
import logging 
from sklearn.model_selection import train_test_split

class Experiment:

    def __init__(self, attributes, classifications):
        ''' Constructor
        '''
        # what data are we looking at
        self._atttributes = attributes
        self._classifications = classifications

    def run(self):
        pass
    
    def _split_train_test(self, test_size=0.3):
        '''Split up the data correctly according to a ratio

        Returns:
            The split data
        '''
        return train_test_split(self._atttributes, self._classifications, \
          test_size=test_size, random_state=0, stratify=self._classifications)

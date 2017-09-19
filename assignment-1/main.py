'''main.py

The driver to the entire assignment
'''
import argparse
import clean
from knn import KNN
from svm import SVM
import logging
import pandas as pd
import os
logging.basicConfig(level=logging.INFO)

CLASSIFIERS = {
  'knn': KNN,
  'svm': SVM
}

if __name__ == '__main__':
    # parse here
    parser = argparse.ArgumentParser(prog='main.py')
    parser.add_argument('-d', '--dataset', help='Which dataset to run on', choices=['wine', 'credit_card'], default='wine')
    subparsers = parser.add_subparsers(title='subcommands', dest='command')
    cleaner_parser = subparsers.add_parser('clean', help='Clean the stats from original to final and show me information')
    knn_parser = subparsers.add_parser('knn', help='Run k-nearest neighbors')
    svm_parser = subparsers.add_parser('svm', help='Run Support Vector Machines')
    args = parser.parse_args()

    # print something out!
    if not args.command:
        parser.print_help()
    
    log = logging.getLogger(__name__)
    command = args.command
    # clean is a one off
    if command == 'clean':
        log.info('Cleaning datasets')  
        clean.create_final_datasets()

    path = './results/{}/{}'.format(args.dataset, command)
    if not os.path.exists(path):
        log.info('Making results directory')
        os.makedirs(path)
    
    if command != 'clean':
        if args.dataset == 'wine':
            log.info('Exploring wine dataset')
            dataset = pd.read_csv('./data/wine-red-white-final.csv')
            class_target = 'red'
            classifications = dataset[class_target]
            attributes = dataset.drop(class_target, axis=1)
        else:
            log.info('Exploring credit card dataset')
            dataset = pd.read_csv('./data/credit-card-final.csv')
            class_target = 'default_payment_next_month'
            classifications = dataset[class_target]
            attributes = dataset.drop(class_target, axis=1)

        log.info('Running %s', command)
        args.classifications = classifications
        args.attributes = attributes
        CLASSIFIERS[command](**vars(args)).run()        

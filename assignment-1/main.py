'''main.py

The driver to the entire assignment
'''
import argparse
import clean
from knn import KNN
import logging
import pandas as pd
logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    # parse here
    parser = argparse.ArgumentParser(prog='main.py')
    parser.add_argument('-d', '--dataset', help='Which dataset to run on', choices=['wine', 'credit_card'], default='wine')
    subparsers = parser.add_subparsers(title='subcommands', dest='command')
    cleaner_parser = subparsers.add_parser('clean', help='Clean the stats from original to final and show me information')
    knn_parser = subparsers.add_parser('knn', help='run k-nearest neighbors')
    knn_parser.add_argument('-n', '--number', type=int, help='Amount of neighbors')
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
    
    if command != 'clean':
        if args.dataset == 'wine':
            log.info('Exploring wine dataset')
            dataset = pd.read_csv('./data/wine-red-white-final.csv')
        else:
            log.info('Exploring credit card dataset')
            dataset = pd.read_csv('./data/credit-card-final.csv')

        if command == 'knn':
            log.info('Running KNN')
            KNN().run()        

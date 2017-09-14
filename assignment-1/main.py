'''main.py

The driver to the entire assignment
'''
import argparse
import clean
from knn import KNN
import logging
logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    # parse here
    parser = argparse.ArgumentParser(prog='main.py')
    subparsers = parser.add_subparsers(title='subcommands', dest='command')
    cleaner_parser = subparsers.add_parser('clean', help='Clean the stats from original to final and show me information')
    knn_parser = subparsers.add_parser('knn', help='run k-nearest neighbors')
    knn_parser.add_argument('-n', '--number', type=int, help='Amount of neighbors')
    args = parser.parse_args()

    # print something out!
    if not args.command:
        parser.print_help()

    command = args.command
    if command == 'clean':
        clean.create_final_datasets()
    elif command == 'knn':
        KNN().run()        

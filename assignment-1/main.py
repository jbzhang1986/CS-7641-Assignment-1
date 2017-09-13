'''main.py

The driver to the entire assignment
'''
import argparse

parser = argparse.ArgumentParser(prog='main.py')
subparsers = parser.add_subparsers(title='subcommands', dest='command')

stats_parser = subparsers.add_parser('stats', help='show some stats about the data')

cleaner_parser = subparsers.add_parser('clean', help='clean the stats from original to final')


knn_parser = subparsers.add_parser('knn', help='run k-nearest neighbors')
knn_parser.add_argument('-n', '--number', type=int, help='Amount of neighbors')
args = parser.parse_args()

# print something out!
if not args.command:
    parser.print_help()

command = args.command
if command == 'cleaner':
elif command == 'stats':
    pass

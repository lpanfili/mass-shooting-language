# Given a csv of tweets (extracted using GetOldTweets) and an output
# file, iterate through the tweets and prompt user for a semicolon-separated
# list of nouns used to describe the shooter in that tweet.
# Each result is written as a new line in the output file, so line numbers
# correspond to the line numbers of tweets in the input file.

import csv
import argparse


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('input', type=str,
						 help='the path to the input data')
	parser.add_argument('output', type=str,
						 help='the path to the output data')
	return parser.parse_args()


def get_nouns(tweet):
	print()
	print(tweet)
	return input("Nouns (semicolon-separated): ")


def main():
	print("For each tweet, type a semicolon-separated list of nouns used" + \
		" to describe the shooter (if none, then just press Enter).")
	print("----------------------------------------")
	print()

	args = parse_args()
	with open(args.output, 'w') as outputFile:
		outputFile.write("Nouns\n")
		with open(args.input, 'r') as inputFile:
			reader = csv.DictReader(inputFile, delimiter=';')
			for line in reader:
				outputFile.write(get_nouns(line['text']))
				outputFile.write('\n')


if __name__ == "__main__":
    main()

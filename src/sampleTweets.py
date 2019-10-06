import argparse
import os
import pandas
import numpy as np

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('tweets_dir', type=str,
						 help='the directory containing tweets by incident')
	parser.add_argument('sample_size', type=int,
						 help='the size of the sample to create')
	parser.add_argument('output_file', type=str,
						 help='the file to contain the sampled tweets')
	return parser.parse_args()

def make_tweets_arr(filenames):
	tweets = []
	for file in filenames:
		num_tweets = pandas.read_csv(file).shape[0]
		tweets.extend([[file, i] for i in range(num_tweets)])
	return tweets

def sample_tweets(tweets_arr, sample_size):
	return np.random.permutation(tweets_arr)[:sample_size]

def write_sample(sampled_tweets, output_file):
	sample = pandas.DataFrame()
	byFile = {}
	for filename, tweet_index in sampled_tweets:
		if filename not in byFile:
			byFile[filename] = []
		byFile[filename].append(int(tweet_index))

	for filename in byFile:
		print("Getting "+str(len(byFile[filename]))+" tweets from "+filename)
		df = pandas.read_csv(filename)
		incident_sample = df.loc[byFile[filename]]
		incident_sample['incident_id'] = os.path.basename(filename).split('.')[0]
		sample = sample.append(incident_sample)

	sample.to_csv(output_file, index=False)

def main():
	args = parse_args()
	for path, dirnames, filenames in os.walk(args.tweets_dir):
		full_names = [os.path.join(path, file) for file in filenames]
		tweets_arr = make_tweets_arr(full_names)
		sampled_tweets = sample_tweets(tweets_arr, args.sample_size)
		write_sample(sampled_tweets, args.output_file)

if __name__ == "__main__":
	main()

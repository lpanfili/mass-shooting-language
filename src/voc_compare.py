import os
import re
import nltk
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('tweets_dir', type=str, help='The directory containing the tweets by incident')
    parser.add_argument('sample_file', type=str, help='The file containing the sampled tweets')
    return parser.parse_args()

def get_all_tweets(tweets_dir):
    dfs = []
    for path, dirnames, filenames in os.walk(tweets_dir):
        for filename in filenames:
            dfs.append(pd.read_csv(os.path.join(path, filename)))
    return pd.concat(dfs)

def get_sample_ids(sample_file):
    return set(pd.read_csv(sample_file)['id'].tolist())

def main():
    args = parse_args()
    tweets = get_all_tweets(args.tweets_dir)
    sample_ids = get_sample_ids(args.sample_file)
    tweets['is_in_sample'] = tweets['id'].isin(sample_ids)
    tweets['text'] = tweets['text'].astype(str)

    tokenizer = nltk.tokenize.TweetTokenizer()
    sample_vectorizer = CountVectorizer(tokenizer=tokenizer.tokenize)
    sample_vectorizer.fit_transform(tweets[tweets['is_in_sample']]['text'])
    sample_voc = sample_vectorizer.get_feature_names()

    sampled_vectorizer = CountVectorizer(vocabulary=sample_voc, max_df=0.95,
                                    min_df=20,
                                    tokenizer=tokenizer.tokenize,
                                    stop_words=ENGLISH_STOP_WORDS)
    tf_sampled = sampled_vectorizer.fit_transform(tweets['text'])
    print(tf_sampled.shape)
    print(np.sum(tf_sampled))
    print()

    full_vectorizer = CountVectorizer(max_df=0.95,
                                    min_df=20,
                                    tokenizer=tokenizer.tokenize,
                                    stop_words=ENGLISH_STOP_WORDS)
    tf_full = full_vectorizer.fit_transform(tweets['text'])
    print(tf_full.shape)
    print(np.sum(tf_full))

    #tf_feature_names = [name.replace(' ', '_') for name in tf_vectorizer.get_feature_names()] # Underscore ngrams

if __name__ == '__main__':
    main()
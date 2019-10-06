from urllib.parse import urlparse
import re
import nltk
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np
from crawler import extract_text_from_url
from extraction_rules import domainHasRules
from getTweets import make_date_incident 
import lda

def flatten(df, column_to_flatten, new_name, split_on):
    s = df[column_to_flatten].str.split(split_on).apply(pd.Series,1).stack()
    s.index = s.index.droplevel(-1)
    s.name = new_name

    del df[column_to_flatten]
    return df.join(s)

def calculateGlobalWeightedLogOddsRatios(doc_classes, doc_word_freqs, alpha0):
    '''
    Given an M by 2 matrix of class assignments
    and an M by W matrix of word frequencies, calculate the weighted
    log-odds ratio (with an informative Dirichlet prior) of each word.
    See (Monroe et al. 2009). 
    '''

    # Create a 2 by W matrix of word occurences for each class
    class_word_freqs = doc_classes.T @ doc_word_freqs
    class_freqs = np.expand_dims(np.sum(class_word_freqs, axis=1), 0).T

    # Find the total number of words in the dataset
    total_words = np.sum(doc_word_freqs)

    # Calculate the informative Dirichlet
    # prior for each word
    word_alphas = np.expand_dims(np.sum(class_word_freqs, axis=0) * (alpha0 / total_words), 0)

    # Calculate the log-odds ratio between classes
    # for each word in the vocabulary
    class_pi_hat = class_word_freqs + word_alphas
    log_odds = np.diff(np.log(class_pi_hat / ((class_freqs + alpha0) - class_pi_hat)), axis=0)

    # Approximate the variances for the log-odds ratios
    variances = np.sum(1 / class_pi_hat, axis=0)

    # Calculate the z-scores for the log-odds ratios
    z_scores = log_odds / np.sqrt(variances)

    return z_scores

def to_one_hot(arr):
    '''
    Given a 1D array of size N, return an N by M one-hot matrix,
    where M is the largest value in the array.
    '''
    return csr_matrix((np.ones(len(arr)), (np.arange(len(arr)), arr))).toarray()

def get_top_features(sample_feature_weights, feature_names, top_n, reverse=False):
    '''
    Given an N by M matrix of N samples and M features, along with an M-dimensional
    array of feature names, return the top_n highest weighted features for each sample.
    '''
    reverse = -1 if reverse else 1
    top = []
    for sample in sample_feature_weights:
        top.append([feature_names[i] for i in sample.argsort()[::reverse][:-top_n - 1:-1]])
    return top


def main():
    mother_jones = pd.read_csv("../raw/motherjones.csv")
    incident_to_race = {make_date_incident(row['Date'])[1]: row['Race'].lower()=='white' for index, row in mother_jones.iterrows()}
    tweets = pd.read_csv("../100K_sample.csv")
    tweets['race'] = tweets['incident_id'].apply(lambda incident: incident_to_race[incident])

    #sources_with_race = mother_jones[["Race", "Sources"]].copy()
    #sources_with_race["Race"] = sources_with_race["Race"].str.lower()
    #sources_with_race = flatten(sources_with_race, "Sources", "Source", "[;,] ?(?=https?://)| and (?=https?://)")

    #races = (sources_with_race["Race"] == "white").astype(int).tolist()
    #documents = [extract_text_from_url(url) if domainHasRules(urlparse(url).netloc) else "" for url in sources_with_race["Source"].tolist()]

    documents = tweets['text']
    races = tweets['race']
    
    url_regex = re.compile(r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)")
    tokenizer = lambda doc: [w for w in re.compile(r"(?u)\b\w\w+\b").findall(doc)]
    names = set(mother_jones['Shooter'].str.lower().str.split().apply(pd.Series,1).stack().tolist())
    places = set([word.replace(',', '') for word in mother_jones['Location'].str.lower().str.split().apply(pd.Series,1).stack().tolist()])
    #tokenizer = nltk.tokenize.TweetTokenizer().tokenize
    tf_vectorizer = CountVectorizer(vocabulary=set(word.lower() for word in nltk.corpus.words.words()),
                                    max_df=0.95,
                                    min_df=20,
                                    ngram_range=(1, 2),
                                    tokenizer=tokenizer,
                                    stop_words=set(ENGLISH_STOP_WORDS).union(names).union(places))
    tf = tf_vectorizer.fit_transform(documents)
    tf_feature_names = [name.replace(' ', '_') for name in tf_vectorizer.get_feature_names()] # Underscore ngrams

    model = lda.LDA(n_topics=20, n_iter=1500, random_state=1)
    model.fit(tf)  # model.fit_transform(X) is also available
    topic_word = model.topic_word_  # model.components_ also works
    n_top_words = 10
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(tf_feature_names)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))

    ratios = np.nan_to_num(calculateGlobalWeightedLogOddsRatios(to_one_hot(races), tf, 100))
    print([word.encode('utf8') for word in get_top_features(ratios, tf_feature_names, 20)[0]])
    print("-------------")
    print([word.encode('utf8') for word in get_top_features(ratios, tf_feature_names, 20, True)[0]])

if __name__ == "__main__":
    main()



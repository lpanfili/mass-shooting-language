import itertools
from urllib.parse import urlparse
import hashlib
import os
from bs4 import BeautifulSoup
import pandas as pd
import requests

from extraction_rules import getArticleSelectorsForDomain

CACHE_DIR = '../data/sources'

def get_document(url, cache_dir, overwrite=False):
    url_hash = hashlib.md5(url.encode('utf8')).hexdigest()
    filename = os.path.join(cache_dir, url_hash + ".html")
    
    if not overwrite and os.path.exists(filename):
        with open(filename, 'r') as input_file:
            return input_file.read()
    else:
        document = requests.get(url).text
        with open(filename, 'w') as output_file:
            output_file.write(document)
        return document

def extract_text_from_url(url):
    return extract_text_from_article_html(url, get_document(url, CACHE_DIR))

def extract_text_from_article_html(url, article_html):
    domain = urlparse(url).netloc
    soup = BeautifulSoup(article_html, "lxml")
    selectors = getArticleSelectorsForDomain(domain)
    if selectors:
        return "\n".join([el.get_text() for el in itertools.chain(
            *[soup.select(selector) for selector in selectors]
        )])
    else: raise Exception("No article text selector(s) for domain " + domain)

def get_mother_jones_urls(data_file):
    data = pd.read_csv(data_file)
    urls = list(itertools.chain(*data["Sources"].str.split("[,;] ?| and ").values))
    return urls

def crawl_mother_jones(data_file):
    urls = get_mother_jones_urls(data_file)
    for url in urls:
        domain = urlparse(url).netloc
        if domain == "www.nytimes.com":
            text = extract_text_from_url(url)
            print(url)
            print((text[:100] + "...").encode('utf8', 'ignore'))
            print("---------------------------------------")



def main():
    crawl_mother_jones('../data/raw/motherjones.csv')

if __name__ == "__main__":
    main()
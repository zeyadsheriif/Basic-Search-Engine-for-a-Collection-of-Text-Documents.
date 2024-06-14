# Data Collection:

#install the Pyterrier framework
!pip install python-terrier
# install the nltk modules
!pip install nltk

# Need to install additional terrier package for PRF. It will take around 1 min
!git clone https://github.com/terrierteam/terrier-prf/
!apt-get install maven   #used for Java projects to manage project dependencies and build processes
%cd /content/terrier-prf/
!mvn install
!pwd
%cd ..

import pyterrier as pt
if not pt.started():
  # In this lab, we need to specify that we start PyTerrier with PRF enabled
  pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])

#Import the necessary modules:
import ipywidgets as widgets
from IPython.display import display
import pandas as pd
import pyterrier as pt
import time
import pandas as pd
import pyterrier as pt
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import *
from nltk.stem.porter import *
import re # used to clean the dat
import os
#to display the full text on the notebook without truncation
pd.set_option('display.max_colwidth', 150)
nltk.download('punkt')

vaswani_dataset = pt.datasets.get_dataset("vaswani")
topics = vaswani_dataset.get_topics()
topics["docno"]=topics["qid"].astype(str)
topics[["docno"]]

qrels = vaswani_dataset.get_qrels()
qrels['docno']=qrels['docno'].astype(str)

topics
qrels

#Preprocessing:

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
print(stopwords.words('english'))

# Initialize Porter stemmer
stemmer = PorterStemmer()

def Steem_text(text):

    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    # print (tokens)
    return ' '.join(stemmed_tokens)

#a function to clean the tweets
def clean(text):
   text = re.sub(r"http\S+", " ", text) # remove urls
   text = re.sub(r"RT ", " ", text) # remove rt
   text = re.sub(r"@[\w]*", " ", text) # remove handles
   text = re.sub(r"[\.\,\#_\|\:\?\?\/\=]", " ", text) # remove special characters
   text = re.sub(r'\t', ' ', text) # remove tabs
   text = re.sub(r'\n', ' ', text) # remove line jump
   text = re.sub(r"\s+", " ", text) # remove extra white space
   text = text.strip()
   return text

def remove_stopwords(text):

    tokens = word_tokenize(text)
    filtered_tokens = [word.lower() for word in tokens if word.lower() not in stop_words] #Lower is used to normalize al the words make them in lower case
    #print('Tokens are:',tokens,'\n')
    return ' '.join(filtered_tokens)

#we need to process the query also as we did for documents
def preprocess(sentence):
  sentence = remove_stopwords(sentence)
  sentence = clean(sentence)
  sentence = Steem_text(sentence)
  return sentence

topics['processed_query'] = topics['query'].apply(preprocess)
topics

topics['processed_query'] = topics['processed_query'].apply(remove_stopwords)
topics

# Display the  processed DataFrames
print('dataFrame after processing:\n')
topics
qrels

#Indexing:

indexref = vaswani_dataset.get_index()
index = pt.IndexFactory.of(indexref)
print(index.getCollectionStatistics().toString())

indexer = pt.DFIndexer("./DatasetIndex", overwrite=True)
index_ref = indexer.index(topics["processed_query"], topics["docno"])
print(index_ref.toString())

index = pt.IndexFactory.of(index_ref)

inverted_index = {}
for doc_id, document in topics.iterrows():
    tokens = word_tokenize(document["processed_query"])

    for token in tokens:
        if token not in inverted_index:
            inverted_index[token] = {}

        if doc_id not in inverted_index[token]:
            inverted_index[token][doc_id] = 0

        inverted_index[token][doc_id] += 1
print("Inverted index:", inverted_index)

term_document_matrix = {}
for term, documents in inverted_index.items():
    term_document_matrix[term] = list(documents.keys())
print("Term-document matrix:", term_document_matrix)


term_frequency_matrix = {}
for term, documents in inverted_index.items():
    term_frequency_matrix[term] = []

    for doc_id, frequency in documents.items():
        term_frequency_matrix[term].append((doc_id, frequency))
print("Term-frequency matrix:", term_frequency_matrix)



#Query Processing:

def query_process(query_text, inverted_index, term_frequency_matrix, topics_data):
    processed_query = preprocess(query_text)
    query_tokens = processed_query.split()
    doc_scores = {}

    for term in query_tokens:
        if term in inverted_index:
            idf = np.log(len(topics_data) / len(inverted_index[term]))
            for doc_id, tf in term_frequency_matrix[term]:
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0
                doc_scores[doc_id] += tf * idf

        ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_docs

query_text = "measurement of dielectric constant of liquids"
ranked_docs = query_process(query_text, inverted_index, term_frequency_matrix, topics)
print("Ranked documents for query '{}':".format(query_text))
for doc_id, score in ranked_docs:
    doc_info = topics.loc[topics['docno'] == str(doc_id)]
    if not doc_info.empty:
        print("Document ID: {}, Query: '{}', Score: {}".format(doc_id, doc_info['query'].values[0], score))
    else:
        print("Document ID: {}, Score: {}".format(doc_id, score))


#Query Expansion:

if not pt.started():
  # In this lab, we need to specify that we start PyTerrier with PRF enabled
  pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])

TFIDF = pt.BatchRetrieve(index, wmodel="TF_IDF",num_results=10)

query="measurement of dielectric constant of liquids"
query = preprocess(query)

expansion_res = TFIDF.search(query)
expansion_res

if not pt.started():
  # In this lab, we need to specify that we start PyTerrier with PRF enabled
  pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])

rm3_expander = pt.rewrite.RM3(index,fb_terms=10, fb_docs=100)

rm3_qe = TFIDF >> rm3_expander
expanded_query = rm3_qe.search(query).iloc[0]["query"]
expanded_query

# Just print the expanded query with term scores
for s in expanded_query.split()[1:]:
  print(s)

print("\n" + query)

# After that you can search using the expanded query
expanded_query_formatted = ' '.join(expanded_query.split()[1:])

results_wqe = TFIDF.search(expanded_query_formatted)

print("   Before Expansion    After Expansion")
print(pd.concat([expansion_res[['docid','score']][0:5].add_suffix('_1'),
            results_wqe[['docid','score']][0:5].add_suffix('_2')], axis=1).fillna(''))

#Let's check the tweets text for the top 5 retrieved tweets
topics[['processed_query']][topics['docno'].isin(results_wqe['docno'].loc[0:5].tolist())]

#Elmo Model:
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

elmo = hub.load("https://tfhub.dev/google/elmo/3")

query = "measurement of dielectric constant of liquids"
query_tokens = preprocess(query)
query_tf = tf.constant([query_tokens])
expansion_res = elmo.signatures["default"](query_tf)["elmo"]
expansion_res



#User Interface:

def search_query(query):
    results = TFIDF.search(query)
    output.clear_output()
    with output:
        for i, (doc_id, score) in enumerate(zip(results['docid'], results['score']), start=1):
            doc_info = topics.loc[topics['docno'] == str(doc_id)]
            if not doc_info.empty:
                print(f"Rank {i}: Document ID: {doc_id}, Query: '{doc_info['query'].values[0]}', Score: {score}")
            else:
                print(f"Rank {i}: Document ID: {doc_id}, Score: {score}")

query_input = widgets.Text(placeholder='Enter your query here...', description='Query:')
query_button = widgets.Button(description="Search")

output = widgets.Output()

def on_button_click(b):
    query = query_input.value
    search_query(query)

query_button.on_click(on_button_click)
display(query_input, query_button, output)


# Evaluation:

def evaluate_search_engine(query, topics, retrieval_model):
    start_time = time.time()
    results = TFIDF.search(query)
    end_time = time.time()
    mean_time = end_time - start_time
    return mean_time

user_query = input ("Enter your query: ")
retrieval_time = evaluate_search_engine(user_query, qrels, TFIDF)
print("Retrieval Time :", retrieval_time, 'seconds')









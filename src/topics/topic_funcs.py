import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gensim
import logging
import io
import os
import re
import operator
import pickle
from shutil import copyfile
from sys import exit
import scipy


from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel

from nltk.collocations import BigramAssocMeasures, TrigramAssocMeasures, QuadgramCollocationFinder, BigramCollocationFinder, TrigramCollocationFinder
from nltk import word_tokenize, FreqDist, bigrams, trigrams

with open('../src/topics/trigrams_filter.txt', 'rb') as f:
    trigrams_filter = pickle.load(f)
with open('../src/topics/bigrams_filter.txt', 'rb') as f:
    bigrams_filter = pickle.load(f)


# For the LDA model, I'd like to filter out the top 3 most frequently used tokens in the corpus
# For most products, the most frequent tokens  refer to the product itself in a generic way 
# and also contain what we might consider stop words
# but didn't want to remove earlier because they are needed for part-of-speech tagging
# (e.g., "product," "coconut," "oil" as well as "have," "be," etc.)
# this helper function is a slight modification of one of gensim's built-in methods -
# I don't want to remove the codewords I've inserted, and those also end up in the top 3 quite often

def remove_freq(dictionary, n):
    """
    Removes the n most frequently appearing terms from the dictionary for topic modeling
    :param object dictionary: the term frequency dictionary
    :param int n: the number of terms to be removed

    This function modifies the dictionary directly and outputs logging information on the
    terms removed from it. 
    """
    logger = logging.getLogger('gensim.corpora.dictionary')
    save = set(['GOODREVIEW', 'BADREVIEW', 'VGOODREVIEW', 'VBADREVIEW'])
    most_frequent_ids = (v for v in (dictionary.token2id).values() if dictionary[v] not in save)
    most_frequent_ids = sorted(most_frequent_ids, key=dictionary.dfs.get, reverse=True)
    most_frequent_ids = most_frequent_ids[:n]
    # do the actual filtering, then rebuild dictionary to remove gaps in ids
    most_frequent_words = [(dictionary[idx], dictionary.dfs.get(idx, 0)) for idx in most_frequent_ids]
    logger.info('discarding %i tokens: %s...', len(most_frequent_ids), most_frequent_words[:10])

    dictionary.filter_tokens(bad_ids=most_frequent_ids)
    logger.info('resulting dictionary: %s', dictionary)


def print_topics(product, df, final_results, input_text, load_path):
    data = df[df['ProductId']==product]
    # prepare the corpus
    texts = data[input_text].str.split()
    dictionary = corpora.Dictionary(texts)
    remove_freq(dictionary, 10)
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    # Load in the tuned LDA model for the product
    # Get the key parameter values required to do this
    t = final_results.loc[product, 'num_topics']
    p = final_results.loc[product, 'passes']
    tn = final_results.loc[product, 'top_n removed'].astype(int)
    na = final_results.loc[product, 'n_above threshold']

    lda = gensim.models.ldamodel.LdaModel.load(".../models/{}/final_models/{}_{}_{}_{}_{}".format(load_path, 
                                                                                         product, 
                                                                                         t, p,
                                                                                         tn, na))
    print('Topic Coherences for Product {}'.format(product))
    # make sure to set the coherence measure appropriately (c_v was used in initial tuning)
    print([lda.top_topics(texts=texts, corpus=corpus, coherence='c_v')][0])
    for topic in range(0,t):
        print('Top Words for Topic {}'.format(topic))
        l = lda.show_topic(topic, topn=20)
        words = [x[0] for x in l]
        print(words)
        print()


def get_topic_data(product, df, final_results, input_text, load_path, encoding_type):
    data = df[df['ProductId']==product]
    # prepare the corpus
    texts = data[input_text].str.split()
    dictionary = corpora.Dictionary(texts)
    remove_freq(dictionary, 10)
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    # Load in the tuned LDA model for the product
    t = final_results.loc[product, 'num_topics']
    p = final_results.loc[product, 'passes']
    tn = final_results.loc[product, 'top_n removed'].astype(int)
    na = final_results.loc[product, 'n_above threshold']

    lda = gensim.models.ldamodel.LdaModel.load('../models/{}/final_models/{}_{}_{}_{}_{}'.format(load_path, 
                                                                                         product, 
                                                                                         t, p,
                                                                                         tn, na))    
    topic_data=[]
    # be sure to set the appropriate coherence measure
    topics = lda.top_topics(texts=texts, corpus=corpus, coherence='c_v')
    
    # iterate through the topics to get coherence, top review, key words, and bigrams
    for topic in range(0,t):
        # sub dataframe where this is the main topic
        main_topic_df = data[data['{} Topic'.format(encoding_type)]==topic] 
        # sub dataframe where this is a subtopic
        sub_topic_df = data[data['{} Subtopic'.format(encoding_type)]==topic]
        # grab the coherence measure
        coherence = (topics[topic][-1]) 
        # Make a list of the top words from the topic
        l = lda.show_topic(topic, topn=10) 
        # And then reformat this into a usable list 
        top_words = [x[0] for x in l]
        # Get the number of reviews fitting into the topic ...
        # as the main topic, with a fit value above 0.7
        as_main = len(main_topic_df.loc[main_topic_df['{} Fit'.format(encoding_type)]>=0.7])
        # as the primary subtopic
        as_primary_sub = len(main_topic_df.loc[(main_topic_df['{} Fit'.format(encoding_type)]<0.7)&
                                               (main_topic_df['{} Fit'.format(encoding_type)]>=0.3)])
        as_secondary_sub = len(sub_topic_df.loc[sub_topic_df['{} Subtopic Fit'.format(encoding_type)]>=0.3])
        #count = len(data[data['{} Topic'.format(encoding_type)]==topic]) 
        try: 
            # Get an index locator for the best fitting review
            ix = main_topic_df['{} Fit'.format(encoding_type)].idxmax(axis=0) 
            # Find the review that best matches the topic
            top_review = main_topic_df.loc[ix, 'clean_review'] 
            # Get that best review's fit value (probability review comes from topic)
            fit = main_topic_df['{} Fit'.format(encoding_type)].max(axis=0) 
            # Getting the bigrams
            bigram_measures = BigramAssocMeasures()
            trigram_measures = TrigramAssocMeasures()
            
            # Build the bigram distribution over the set of words found in the reviews tagged to this topic
            #words = np.concatenate(np.array([word_tokenize(r) for r in sub_df['{}_x'.format(input_text)].values])) 
            words = np.concatenate(np.array([word_tokenize(r) for r in main_topic_df['clean_vanilla_x'].values])) 

            bigram_fd = FreqDist(bigrams(words))
            trigram_fd = FreqDist(trigrams(words))

            bfinder = BigramCollocationFinder.from_words(words, window_size=3)
            tfinder = TrigramCollocationFinder.from_words(words, window_size=4)
            for finder in [bfinder, tfinder]:
                # Get rid of words we don't want
                finder.apply_word_filter(lambda w: w in ('GOODREVIEW', 'BADREVIEW', 
                                                         'VGOODREVIEW', 'VBADREVIEW', 
                                                         's', 'b', 'c', 'oz', 'be')) 
                                                         
                # Filter out bigrams that don't appear at least 2 times
                finder.apply_freq_filter(2) 
                
            # Filter out some common n-grams
            bfinder.apply_ngram_filter(lambda w1, w2: (w1, w2) in bigrams_filter) 
            tfinder.apply_ngram_filter(lambda w1, w2, w3: (w1, w2, w3) in trigrams_filter)
            # Get the top 3 bigrams and trigrams by raw frequency and by PMI value
            bgrams_pmi = bfinder.nbest(bigram_measures.pmi, 10) 
            bgrams_freq = bfinder.nbest(bigram_measures.raw_freq, 10) 
            tgrams_pmi = tfinder.nbest(trigram_measures.pmi, 10)
            tgrams_freq = tfinder.nbest(trigram_measures.raw_freq, 10)
            # Format a bit more nicely for readability
            top_bigrams_pmi = [a[0]+" "+a[1] for a in bgrams_pmi] 
            top_bigrams_freq = [a[0]+" "+a[1] for a in bgrams_freq[2:]]
            top_trigrams_pmi = [a[0]+" "+a[1]+" "+a[2] for a in tgrams_pmi]
            top_trigrams_freq = [a[0]+" "+a[1]+" "+a[2] for a in tgrams_freq[2:]]

        except ValueError: 
            # ValueError in this case indicates there were no reviews that were matched to the topic
            # hence the results will be blank for that
            top_review = 'none'
            fit = ''
            top_bigrams_pmi = []
            top_trigrams_pmi = []
            top_bigrams_freq = []
            top_trigrams_freq = []
   
        
        topic_data.append([product, topic, as_main,
                           as_primary_sub, as_secondary_sub, 
                           coherence, top_words, 
                           top_review, fit, top_bigrams_pmi, 
                           top_bigrams_freq, top_trigrams_pmi, 
                           top_trigrams_freq])
                
        
    topic_data=pd.DataFrame(data=topic_data, 
                            columns=['product', 'topic', 'as_main_topic',
                                     'as_primary_subtopic', 'as_secondary_subtopic',
                                    'topic_coherence', 'top_words', 
                                    'best_review', 'best_review_fit', 
                                    'top_bigrams_pmi', 'top_bigrams_freq',
                                    'top_trigrams_pmi', 'top_trigrams_freq'])
    return topic_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gensim
import logging
import io
import os
import re
import itertools
from shutil import copyfile
from sys import exit
from six import itervalues


from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel


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
    This is a modified version of the gensim Dictionary method of the same name.
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

# I also need a method that can remove tokens that appear in fewer than n reviews, or keep only those that 
# appear in more than x% of the reviews. This is also a modification of the gensim Dictionary method
# of the same name that allows me to avoid dropping the codewords.

def filter_extremes(dictionary, no_above, no_below, keep_n=100000):
    """
    Filter out tokens that appear in

    1. less than `no_below` documents (absolute number) or
    2. more than `no_above` documents (fraction of total corpus size, *not*
       absolute number).
    3. after (1) and (2), keep only the first `keep_n` most frequent tokens (or
       keep all if `None`).

    After the pruning, shrink resulting gaps in word ids.

    **Note**: Due to the gap shrinking, the same word may have a different
    word id before and after the call to this function!
    """
    logger = logging.getLogger('gensim.corpora.dictionary')

    no_above_abs = int(no_above * dictionary.num_docs)  # convert fractional threshold to absolute threshold

    # determine which tokens to keep
    code_ids = []
    for t in ['GOODREVIEW', 'BADREVIEW', 'VGOODREVIEW', 'VBADREVIEW']:
        try:
            code_ids.append(dictionary.token2id[t])
        except KeyError:
            continue

    good_ids = (
        v for v in itervalues(dictionary.token2id)
        if no_below <= dictionary.dfs.get(v, 0) <= no_above_abs
        or v in code_ids)
    good_ids = sorted(good_ids, key=dictionary.dfs.get, reverse=True)
    if keep_n is not None:
        good_ids = good_ids[:keep_n]
    bad_words = [(dictionary[id], dictionary.dfs.get(id, 0)) for id in set(dictionary).difference(good_ids)]
    logger.info('discarding %i tokens: %s...', len(dictionary) - len(good_ids), bad_words[:10])
    logger.info(
        'keeping %i tokens which were in no less than %i and no more than %i (=%.1f%%) documents',
        len(good_ids), no_below, no_above_abs, 100.0 * no_above)

    # Do the actual filtering, then rebuild dictionary to remove gaps in ids
    dictionary.filter_tokens(good_ids=good_ids)
    logger.info('resulting dictionary: %s', dictionary)


# the next few helper functions deal with extracting metrics of interest from gensim's logger, 
# which are being dumped into a log file as training runs
# I am capturing bounds, perplexity, and per-word topic differences
    
    
# thanks to these SO answers https://stackoverflow.com/questions/6213063/python-read-next
# and https://stackoverflow.com/questions/5434891/iterate-a-list-as-pair-current-next-in-python
# for showing a way to deal with lines of a file in groups of three
def threes(iterator):
    's -> (s0,s1,s2), (s1,s2,s3), (s2, s3, s4), ...'
    a, b, c = itertools.tee(iterator, 3)
    next(b, None)
    next(c, None)
    next(c, None)
    return zip(a, b, c)

def capture_logs(): 
# capture the perplexity, per-word bound, and topic difference values from the logger and save
    perplexity_log = []
    perplex = {}
    bounds = {}
    diff = {}
    with open('../models/training_output.log', 'r') as f:
        for line in f:
            if re.match("|".join([r'.*topic diff.*', r'.*per-word.*', r'.*PROGRESS.*']), line):
                perplexity_log.append(line)
    for a, b, c in threes(perplexity_log): 
        if re.match(r'.*PROGRESS.*', a):
            pass_val = int(re.search(r'\d*, at', a).group(0).split(',')[0])
            if re.match(r'.*topic diff.*', b):
                d = float(re.search(r'\d*\.\d*', b).group(0).split()[0])
                diff[pass_val] = d
            if re.match(r'.*per-word.*', b): # these may show up in the second line of the group as well
                b = float(re.search(r'.\d*\.\d* per', b).group(0).split()[0])
                p = float(re.search(r'\d*\.\d perplexity', b).group(0).split()[0])
                perplex[pass_val] = p
                bounds[pass_val] = b
            if re.match(r'.*per-word.*', c):
                b = float(re.search(r'.\d*\.\d* per', c).group(0).split()[0])
                p = float(re.search(r'\d*\.\d perplexity', c).group(0).split()[0])
                perplex[pass_val] = p
                bounds[pass_val] = b
    return bounds, perplex, diff

def perplexity_decreasing(perplex): 
#checks if the perplexity decreased during the last two training passes
    passes = sorted(perplex.keys())
    start = passes[-2]
    end = passes[-1]
    if perplex[start] > perplex[end]:
        return True
    else:
        return False


def run_lda(df, product, n_topics, n_passes, texts, save_path):
    """
    This function trains an LDA model fo a single product, 
    constructing any number of topics over any number of training passes.

    :param DataFrame df: the input data
    :param str product: the string product ID
    :param int n_topics: number of topics desired
    :param int n_passes: number of training passes to make
    :param str texts: the corpus to use (column name from the main dataframe)
    :param str save_path: where to save the model outputs (depending on the type of text encoding used)
    :return: a dataframe with the results of the training
    :rtype: DataFrame 
    """
    os.remove('../models/training_output.log') # clear the log file
    logger = logging.getLogger('gensim.models.ldamodel')
    handler = logging.FileHandler('../models/training_output.log')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s : %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    data = df[df['ProductId']==product]
    texts = data[texts].str.split()
    dictionary = corpora.Dictionary(texts)
    remove_freq(dictionary, 3) # remove the 3 most frequently appearing tokens
    corpus = [dictionary.doc2bow(text) for text in texts]
    corpora.MmCorpus.serialize('/tmp/corpus.mm', corpus)
    mm = corpora.MmCorpus('/tmp/corpus.mm')
    review_counts = df['ProductId'].value_counts().sort_values()
    chunk_size = review_counts[product]/3
    lda = gensim.models.ldamodel.LdaModel(corpus=mm, id2word=dictionary, 
                                          num_topics=n_topics, update_every=1, 
                                          chunksize=chunk_size, passes=n_passes)
    bounds, perplex, diff = capture_logs()
    results = pd.DataFrame(index=[product], data={'num_topics': n_topics, 
                                                  'chunk': chunk_size, 
                                                  'passes': n_passes})
    results['per-word bounds'] = [bounds]
    results['perplexity'] = [perplex]
    results['topic diff'] = [diff]
    p = sorted(perplex.keys())
    end = p[-1]
    results['final perplexity'] = perplex[end]
    d = sorted(diff.keys())
    end = p[-1]
    results['final topic diff'] = diff[end]
    if perplexity_decreasing:
        results['perplexity decreasing'] = True
    else: 
        results['perplexity_decreasing'] = False
    for n in range(0,n_topics):
        topic = lda.show_topic(n, 20)
        results['topic {}'.format(n)] = [topic]
    lda.save('../models/{}/{}'.format(save_path, product))
    lda.clear()
    return results


def tune_lda(df, product, n_topics, 
             n_passes, input_text, 
             save_path, n_below, 
             top_n, n_above):
    """
    Tunes an LDA model for a single product by grid searching over different combinations
    of number of topics and number of training passes, using a specified text set 
    (vanilla, coded, or valence coded)

    :param DataFrame df: the main input data in a DataFrame
    :param str product: the product ID as a string
    :param list n_topics: a list of number of topics to be grid searched
    :param list n_passes: a list of number of training passes to be grid searched
    :param str input_text: the type of encoding to be used which corresponds to a column name
                            in the df ("clean_words", "clean_coded", "clean_valence")
    :param str save_path: the save path for the LDA model ("vanilla_outputs", "coded_outputs,
                                    "valence_outputs")
    :return: output, a dataframe containing hyperparamters and performance results for 
            each LDA model constructed by grid search
    :rtype: DataFrame
    """
    os.remove('../models/training_output.log')
    logger = logging.getLogger('gensim.models.ldamodel')
    handler = logging.FileHandler('../models/training_output.log')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s : %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    data = df[df['ProductId']==product]
    texts = data[input_text].str.split()
    output = pd.DataFrame(columns=['product', 'num_topics', 
                                   'chunk', 'passes', 
                                   'per-word bounds', 'perplexity', 
                                   'topic diff', 'final perplexity', 
                                   'final topic diff', 'perplexity decreasing', 
                                   'coherence', 'top_n removed', 
                                   'n_above threshold'])
    for tn in top_n:
        for na in n_above:

            for t in n_topics:
                for p in n_passes:
                    dictionary = corpora.Dictionary(texts)

                    remove_freq(dictionary, tn) #remove the top_n most common tokens

                    filter_extremes(dictionary=dictionary, no_below=n_below, no_above=na) 
                    #remove tokens that show up in n_below or fewer documents, 
                    # and in no_above or more (float %) of documents
                    corpus = [dictionary.doc2bow(text) for text in texts]
                    corpora.MmCorpus.serialize('/tmp/corpus.mm', corpus)
                    mm = corpora.MmCorpus('/tmp/corpus.mm')
                    review_counts = df['ProductId'].value_counts().sort_values()
                    chunk_size = review_counts[product]/3

                    print('training LDA with {} topics over {} passes'.format(t, p))
                    print('removing top {} words with {} review freq threshold'.format(tn, na))
                    lda = gensim.models.ldamodel.LdaModel(corpus=mm, id2word=dictionary, 
                                                          num_topics=t, update_every=1, 
                                                          chunksize=chunk_size, passes=p, 
                                                          random_state=42)
                    bounds, perplex, diff = capture_logs()
                    results = {'product': product, 'num_topics': t, 'chunk': chunk_size, 'passes': p}
                    results['per-word bounds'] = [bounds]
                    results['perplexity'] = [perplex]
                    results['topic diff'] = [diff]
                    per = sorted(perplex.keys())
                    end = per[-1]
                    results['final perplexity'] = perplex[end]
                    d = sorted(diff.keys())
                    end = d[-1]
                    results['final topic diff'] = diff[end]
                    if perplexity_decreasing:
                        results['perplexity decreasing'] = True
                    else: 
                        results['perplexity_decreasing'] = False
                    for n in range(0,t):
                        topic = lda.show_topic(n, 20)
                        results['topic {}'.format(n)] = [topic]

                    lda.save('../models/{}/{}_{}_{}_{}_{}'.format(save_path, product, t, p, tn, na))
                    cm = CoherenceModel(model=lda, corpus=corpus, 
                                        texts = texts, coherence='c_v')
                    results['coherence'] = cm.get_coherence()
                    results['top_n removed'] = tn
                    results['n_above threshold'] = na
                    output = pd.concat([output, pd.DataFrame(data=results)], axis=0)
                    lda.clear()
    return output


def save_best(output, final_results, save_path):
    """
    Finds the best result (maximizing coherence) from the grid search run, saves the information
    on that model into a final results dataframe, and saves off the best LDA model to a separate folder

    :param DataFrame output: the dataframe with outputs from a grid search run
    :param DataFrame final_results: a dataframe for storing information on the best LDA model
                                    including its relevant hyperparameters
    :param str save_path: a string referencing the folder for storing the best model from 
                            the grid search
    :return: final_results, a dataframe with hyperparameters and performance stats on the best models
    :rtype: DataFrame
    """
    output.reset_index(inplace=True)
    best_idx = output['coherence'].idxmax()
    product = output.loc[best_idx, 'product']
    print('best results for product {}:'.format(product))
    print(output.loc[best_idx])
    final_results = final_results.append(output.loc[best_idx], ignore_index=True)
    t = output.loc[best_idx, 'num_topics']
    p = output.loc[best_idx, 'passes']
    tn = output.loc[best_idx, 'top_n removed']
    na = output.loc[best_idx, 'n_above threshold']
    lda = gensim.models.ldamodel.LdaModel.load("../models/{}/{}_{}_{}_{}_{}".format(save_path, product, t, p, tn, na))
    lda.save('../models/{}/final_models/{}_{}_{}_{}_{}'.format(save_path, product, t, p, tn, na))
    lda.clear()
    del output # don't need this interim df anymore
    print('Final model saved for product {} with {} topics over {} passes, removing top {} tokens and token review threshold {}.'.format(product, t, p, tn, na))
    return final_results
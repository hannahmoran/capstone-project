import pandas as pd 
import numpy as np 
import gensim
from PIL import Image
import logging

from gensim import corpora
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from six import itervalues



def remove_key_tokens(dictionary, n, no_above, no_below=0, keep_n=100000):
    """
    Creates a list of custom stopwords for the wordcloud - ingests a gensim corpora dictionary
    from an LDA model and removes the top n most frequent appearing words, as well as words
    that appear in more than no_abov (%) of the reviews. 
    :param object dictionary: the term frequency dictionary for the LDA model (from gensim)
    :param int n: how many of the most frequent terms to remove
    :param float no_abov: the percentage of reviews in which a term must appear to be removed

    This function does not modify the dictionary, but just returns a list of words to be used
    as stopwords by the wordcloud. 

    """
    save = set(['GOODREVIEW', 'BADREVIEW', 'VGOODREVIEW', 'VBADREVIEW'])
    most_frequent_ids = (v for v in (dictionary.token2id).values() if dictionary[v] not in save)
    most_frequent_ids = sorted(most_frequent_ids, key=dictionary.dfs.get, reverse=True)
    most_frequent_ids = most_frequent_ids[:n]
    # do the actual filtering, then rebuild dictionary to remove gaps in ids
    most_frequent_words = [dictionary[idx] for idx in most_frequent_ids]
    #print('discarding %i tokens: %s...', len(most_frequent_ids), most_frequent_words[:10])


    no_above_abs = int(no_above * dictionary.num_docs)  # convert fractional threshold to absolute threshold

    
    code_ids = []
    # add in any code IDs that appear in the dictionary to a list to save
    for t in ['GOODREVIEW', 'BADREVIEW', 'VGOODREVIEW', 'VBADREVIEW']:
        try:
            code_ids.append(dictionary.token2id[t])
        except KeyError:
            continue

    # keep IDs if they pass the threshold or are part of the code ID list for this dict
    good_ids = (
        v for v in itervalues(dictionary.token2id)
        if no_below <= dictionary.dfs.get(v, 0) <= no_above_abs
        or v in code_ids)
    good_ids = sorted(good_ids, key=dictionary.dfs.get, reverse=True)
    if keep_n is not None:
        good_ids = good_ids[:keep_n]
    bad_words = [dictionary[id] for id in set(dictionary).difference(good_ids)]
    remove_words = list(set(bad_words) | set(most_frequent_words))
    
    # toggle this line on or off to remove or keep the codewords in the wordclouds
    #remove_words.extend(['GOODREVIEW', 'BADREVIEW', 'VGOODREVIEW', 'VBADREVIEW'])
    
    #print('discarding %i tokens: %s...', len(dictionary) - len(good_ids), bad_words[:10])
    #print('keeping %i tokens which were in no less than %i and no more than %i (=%.1f%%) documents',
    #    len(good_ids), no_below, no_above_abs, 100.0 * no_above)
    return remove_words

def get_wordcloud_removewords(df, final_results, product, input_text):
    """
    Returns a list of custom stopwords for the wordcloud afer identifying how and which words
    should be removed.
    :param df DataFrame: the full dataframe of reviews
    :param final_results DataFrame: the dataframe specifying the tuned LDA model parameters
    :param product str: the product to make wordclouds for
    :param input_text str: the type of input (vanilla, coded, valence coded)

    This function returns a list of words to be used as stopwords by the wordcloud
    """
    tn = final_results.loc[product, 'top_n removed']
    na = final_results.loc[product, 'n_above threshold']
    data = df[df['ProductId']==product]
    texts = data[input_text].str.split()
    dictionary = corpora.Dictionary(texts)
    remove_words = remove_key_tokens(dictionary, n=tn, no_above=na)
    return remove_words

def generate_wordcloud(df, final_results, product, topic, encoding_type, input_text):
    """
    Generates a wordcloud for a particular product, topic, input text combination. 
    Word frequency is determined by how often the word appears in the actual reviews
    for this product and topic combination. 
    :param df DataFrame: the full dataframe of reviews
    :param final_results DataFrame: the dataframe specifying the tuned LDA model parameters
    :param product str: the product to make wordclouds for
    :param topic int: the topic number
    :param encoding_type: the type of text input
    :param input_text str: the type of input (vanilla, coded, valence coded)

    This function returns a wordcloud image
    """

    # filters out products where the fit value is below 0.7
    data = df.loc[(df['ProductId']==product)&
                  (df['{} Topic'.format(encoding_type)]==topic)&
                  (df['{} Fit'.format(encoding_type)]>=0.7)]
    if len(data>0):
        # only make a wordcloud if there were a non-zero number of reviews in this combination
        remove_words = get_wordcloud_removewords(df, final_results, product, input_text)
        texts = " ".join(review for review in data[input_text])
        wordcloud = WordCloud(stopwords=remove_words,
                             background_color='white',
                             max_words=100,
                             width=1000, height=500,
                             random_state=42).generate(texts)
        return wordcloud
    else:
        print('Error - No wordcloud')
        return None


def get_params(df, product, encoding_type):
    """
    Gets the parameters of the tuned LDA model for a particular product-topic-encoding combination.
    These are necessary to determine which token IDs to remove from the dict for the wordcloud. 
    :param df DataFrame: the full dataframe of reviews
    :param product str: the product to make wordclouds for
    :param encoding_type: the type of text input (Vanilla, Coded, Valence Coded)

    This function returns the a tuple of a list of topics, and the number of topics in the tuned LDA model
    """
    data = df.loc[(df['ProductId']==product)&
                  (df['{} Fit'.format(encoding_type)]>=0.7)]
    topics = list(data['{} Topic'.format(encoding_type)].unique())
    n_topics=len(topics)
    return topics, n_topics

def do_plot(fig, df, product, t, topic_list, encoding_type, gap, final_results, input_text, plot_column, spec):
    data_t = df.loc[(df['ProductId']==product)&
                    (df['{} Topic'.format(encoding_type)]==topic_list[t])&
                    (df['{} Fit'.format(encoding_type)]>=0.7)]

    ax = fig.add_subplot(spec[t+gap+1, plot_column])
    cloud = generate_wordcloud(df=df, final_results=final_results, product=product, 
                               topic=topic_list[t], encoding_type=encoding_type, 
                               input_text=input_text)
    ax.imshow(cloud, interpolation='bilinear')

    ax.set_title("Topic {}, {} reviews".format(topic_list[t], len(data_t)))
    ax.axis('off')
    

    
def make_all_wordclouds(product, df, vanilla_final_results,
						coded_final_results, valence_final_results):
    """
    Plots all wordclouds for a product, using the words from reviews to build the wordclouds vs.
    the topic model word distribution.
    :param df DataFrame: the full dataframe of reviews
    :param product str: the product to make wordclouds for
    :param vanilla_final_results DataFrame: dataframe of tuned vanilla LDA model information
    :param coded_final_results DataFrame: dataframe of tuned coded LDA model information
    :param valence_final_results DataFrame: dataframe of tuned valence LDA model information


    This function plots a grid of wordclouds.
    """

    vanilla_topics, n_vanilla_topics = get_params(df, product, 'Vanilla')
    coded_topics, n_coded_topics = get_params(df, product, 'Coded')
    valence_topics, n_valence_topics = get_params(df, product, 'Valence')

    max_topics = max(n_vanilla_topics, n_coded_topics, n_valence_topics)
    vanilla_gap = max_topics - n_vanilla_topics
    coded_gap = max_topics - n_coded_topics
    valence_gap = max_topics - n_valence_topics
    
    # Heights ratio controls relative height of subplots
    # I'm using a blank row of plots at the top to label the columns
    heights = [0.02]
    heights.extend([1]*max_topics)
    if max_topics>5:
        fig = plt.figure(figsize=[15, 3*max_topics])
    else:
        fig = plt.figure(figsize=[15, 15])
    spec = gridspec.GridSpec(ncols=3, nrows=max_topics+1, figure=fig, wspace=0.1, hspace=0.1,
                            height_ratios=heights)
    

    fig.suptitle("Wordclouds for Product {}".format(product), fontsize=14, fontweight='bold')
    ax_vanilla = fig.add_subplot(spec[0, 0])
    ax_vanilla.set_title("Vanilla Inputs", fontsize=12, fontweight='bold')
    ax_vanilla.axis('off')
    ax_coded = fig.add_subplot(spec[0, 1])
    ax_coded.set_title("Coded Inputs",  fontsize=12, fontweight='bold')
    ax_coded.axis('off')
    ax_valence = fig.add_subplot(spec[0, 2])
    ax_valence.set_title("Valence Inputs",  fontsize=12, fontweight='bold')
    ax_valence.axis('off')

    for t in range(n_vanilla_topics):
        do_plot(fig=fig, df=df, product=product, t=t, topic_list=vanilla_topics, 
                encoding_type='Vanilla', gap=vanilla_gap, 
                final_results=vanilla_final_results, 
                input_text='vanilla_stripped', plot_column=0, spec=spec)
    
    for t in range(n_coded_topics):
        do_plot(fig=fig, df=df, product=product, t=t, topic_list=coded_topics, 
                encoding_type='Coded', gap=coded_gap, 
                final_results=coded_final_results, 
                input_text='coded_stripped', plot_column=1, spec=spec)
        
    for t in range(n_valence_topics):
        do_plot(fig=fig, df=df, product=product, t=t, topic_list=valence_topics, 
                encoding_type='Valence', gap=valence_gap, 
                final_results=valence_final_results, 
                input_text='valence_stripped', plot_column=2, spec=spec)
    
    # Fix up the layout - this removes extra whitespace after the supertitle
    spec.tight_layout(fig, rect=[0, 0.03, 1, 0.97])
    


def make_model_wordclouds_review_counts(product, df, vanilla_final_results,
										coded_final_results, valence_final_results):
    """
    Plots all wordclouds for a product, using the topic model term distribution as the frequency dictionary,
    but filtering to topics with at least one matching review and including the review counts. 
    :param df DataFrame: the full dataframe of reviews
    :param product str: the product to make wordclouds for
    :param vanilla_final_results DataFrame: dataframe of tuned vanilla LDA model information
    :param coded_final_results DataFrame: dataframe of tuned coded LDA model information
    :param valence_final_results DataFrame: dataframe of tuned valence LDA model information


    This function plots a grid of wordclouds.
    """
    vanilla_dicts = get_frequency_dicts(df, vanilla_final_results, product, 'clean_vanilla', 'vanilla_outputs')
    coded_dicts = get_frequency_dicts(df, coded_final_results, product, 'clean_coded', 'coded_outputs')
    valence_dicts = get_frequency_dicts(df, valence_final_results, product, 'clean_valence', 'valence_outputs')
    
    vanilla_topics, n_vanilla_topics = get_params(df, product, 'Vanilla')
    coded_topics, n_coded_topics = get_params(df, product, 'Coded')
    valence_topics, n_valence_topics = get_params(df, product, 'Valence')

    max_topics = max(n_vanilla_topics, n_coded_topics, n_valence_topics)
    vanilla_gap = max_topics - n_vanilla_topics
    coded_gap = max_topics - n_coded_topics
    valence_gap = max_topics - n_valence_topics
    
    # Heights ratio controls relative height of subplots
    # I'm using a blank row of plots at the top to label the columns
    heights = [0.02]
    heights.extend([1]*max_topics)
    if max_topics>5:
        fig = plt.figure(figsize=[15, 3*max_topics])
    else:
        fig = plt.figure(figsize=[15, 12])
    spec = gridspec.GridSpec(ncols=3, nrows=max_topics+1, figure=fig, wspace=0.1, hspace=0.1,
                            height_ratios=heights)
    

    fig.suptitle("Wordclouds for Product {}".format(product), fontsize=14, fontweight='bold')
    ax_vanilla = fig.add_subplot(spec[0, 0])
    ax_vanilla.set_title("Vanilla Inputs", fontsize=12, fontweight='bold')
    ax_vanilla.axis('off')
    ax_coded = fig.add_subplot(spec[0, 1])
    ax_coded.set_title("Coded Inputs",  fontsize=12, fontweight='bold')
    ax_coded.axis('off')
    ax_valence = fig.add_subplot(spec[0, 2])
    ax_valence.set_title("Valence Inputs",  fontsize=12, fontweight='bold')
    ax_valence.axis('off')

    for t in range(n_vanilla_topics):
        topic = vanilla_topics[t]
        ax = fig.add_subplot(spec[t+vanilla_gap+1, 0])
        cloud = WordCloud(background_color='white',
                         random_state=42,
                        width=1000, height=500,).generate_from_frequencies(vanilla_dicts[topic])
        n_reviews = len(df.loc[(df['ProductId']==product)&
                               (df['Vanilla Topic']==vanilla_topics[t])&
                               (df['Vanilla Fit']>=0.7)])
        ax.imshow(cloud, interpolation='bilinear')
        ax.set_title("Topic {}, {} reviews".format(vanilla_topics[t], n_reviews))
        ax.axis('off')
    
    for t in range(n_coded_topics):
        topic = coded_topics[t]
        ax = fig.add_subplot(spec[t+coded_gap+1, 1])
        cloud = WordCloud(background_color='white',
                         random_state=42,
                        width=1000, height=500,).generate_from_frequencies(coded_dicts[topic])
        n_reviews = len(df.loc[(df['ProductId']==product)&
                       (df['Coded Topic']==coded_topics[t])&
                       (df['Coded Fit']>=0.7)])
        ax.imshow(cloud, interpolation='bilinear')
        ax.set_title("Topic {}, {} reviews".format(coded_topics[t], n_reviews))
        ax.axis('off')
        
    for t in range(n_valence_topics):
        topic = valence_topics[t]
        ax = fig.add_subplot(spec[t+valence_gap+1, 2])
        cloud = WordCloud(background_color='white',
                         random_state=42,
                          width=1000, height=500,).generate_from_frequencies(valence_dicts[topic])
        n_reviews = len(df.loc[(df['ProductId']==product)&
                       (df['Valence Topic']==valence_topics[t])&
                       (df['Valence Fit']>=0.7)])
        ax.imshow(cloud, interpolation='bilinear')
        ax.set_title("Topic {}, {} reviews".format(valence_topics[t], n_reviews))
        ax.axis('off')    
    # Fix up the layout - this removes extra whitespace after the supertitle
    spec.tight_layout(fig, rect=[0, 0.03, 1, 0.97])

def get_frequency_dicts(df, final_results, product, input_text, load_path):
    """
    Gets frequency dictionaries for a particular product and input text combination.
    :param df DataFrame: the full dataframe of reviews
    :param product str: the product to make wordclouds for
    :param final_results DataFrame: dataframe of tuned LDA model information
    :param input_text str: name of column of final_results to use to build frequency dictionary
    :param load_path str: where to get the tuned LDA model from


    This function returns a dictionary of dictionaries - one word frequency dictionary for each topic
    belonging to the product and input text type, according to the tuned LDA model. 
    """
    # Load in the tuned LDA model for the product and input text combo
    t = final_results.loc[product, 'num_topics']
    p = final_results.loc[product, 'passes']
    tn = final_results.loc[product, 'top_n removed'].astype(int)
    na = final_results.loc[product, 'n_above threshold']

    lda = gensim.models.ldamodel.LdaModel.load('../models/{}/final_models/{}_{}_{}_{}_{}'.format(load_path, 
                                                                                         product, 
                                                                                         t, p,
                                                                                      tn, na))
    freq_dicts = {}
    for topic in range(0,t):
        freqs = lda.show_topic(topic, topn=100)
        # the above returns a list of tuples of the top n words in the topic
        freq_dict = dict(freqs) 
        freq_dicts[topic] = freq_dict
    return freq_dicts

def do_model_plot(freq_dicts, t, topic_list, gap, plot_column, fig):
    """
    Plots a single wordcloud for a topic and input text type.
    :param freq_dicts dict: a dict of dictionaries, each one containing word frequencies for a topic
    :param t int: the index position of the topic in the topic list
    :param topic_list list: the list of topics
    :param gap int: the difference between the number of topics in this topic list and the max # of topics for the product
    :param plot_column int: the column position for this plot in the figure grid
    :param spec GridSpec: the matplotlib gridspec for this figure


    This function plots a single wordcloud into a grid.
    """
    ax = fig.add_subplot(spec[t+gap+1, plot_column])
    cloud = WordCloud(background_color='white',
                     random_state=42).generate_from_frequencies(freq_dicts[t])
    ax.imshow(cloud, interpolation='bilinear')
    ax.set_title("Topic {}".format(topic_list[t]))
    ax.axis('off')
    
def make_model_wordclouds(product, df, vanilla_final_results,
							coded_final_results, valence_final_results):
    """
    Plots all model wordclouds for a product,  meaning the frequency dictionaries of the topic models
    are what drives the visual
    :param df DataFrame: the full dataframe of reviews
    :param product str: the product to make wordclouds for
    :param vanilla_final_results DataFrame: dataframe of tuned vanilla LDA model information
    :param coded_final_results DataFrame: dataframe of tuned coded LDA model information
    :param valence_final_results DataFrame: dataframe of tuned valence LDA model information


    This function plots a grid of wordclouds.
    """

    vanilla_dicts = get_frequency_dicts(df, vanilla_final_results, product, 'clean_vanilla', 'vanilla_outputs')
    coded_dicts = get_frequency_dicts(df, coded_final_results, product, 'clean_coded', 'coded_outputs')
    valence_dicts = get_frequency_dicts(df, valence_final_results, product, 'clean_valence', 'valence_outputs')

    vanilla_topics = list(vanilla_dicts.keys())
    n_vanilla_topics = len(vanilla_topics)
    
    coded_topics = list(coded_dicts.keys())
    n_coded_topics = len(coded_topics)
    
    valence_topics = list(valence_dicts.keys())
    n_valence_topics = len(valence_topics)
    

    max_topics = max(n_vanilla_topics, n_coded_topics, n_valence_topics)

    vanilla_gap = max_topics - n_vanilla_topics
    coded_gap = max_topics - n_coded_topics
    valence_gap = max_topics - n_valence_topics
    
    # Use heights ratio to control relative height of subplots
    # Create a blank row of plots at the top to label the columns
    heights = [0.02]
    heights.extend([1]*max_topics)
    if max_topics>5:
        fig = plt.figure(figsize=[15, 3*max_topics])
    else:
        fig = plt.figure(figsize=[15, 15])
    spec = gridspec.GridSpec(ncols=3, nrows=max_topics+1, figure=fig, wspace=0.1, hspace=0.1,
                            height_ratios=heights)
    

    fig.suptitle("Topic Model Wordclouds for Product {}".format(product), fontsize=14, fontweight='bold')
    ax_vanilla = fig.add_subplot(spec[0, 0])
    ax_vanilla.set_title("Vanilla Inputs", fontsize=12, fontweight='bold')
    ax_vanilla.axis('off')
    ax_coded = fig.add_subplot(spec[0, 1])
    ax_coded.set_title("Coded Inputs",  fontsize=12, fontweight='bold')
    ax_coded.axis('off')
    ax_valence = fig.add_subplot(spec[0, 2])
    ax_valence.set_title("Valence Inputs",  fontsize=12, fontweight='bold')
    ax_valence.axis('off')

    for t in range(n_vanilla_topics):
        ax = fig.add_subplot(spec[t+vanilla_gap+1, 0])
        cloud = WordCloud(background_color='white',
                         random_state=42).generate_from_frequencies(vanilla_dicts[t])
        ax.imshow(cloud, interpolation='bilinear')
        ax.set_title("Topic {}".format(vanilla_topics[t]))
        ax.axis('off')

    
    for t in range(n_coded_topics):
        ax = fig.add_subplot(spec[t+coded_gap+1, 1])
        cloud = WordCloud(background_color='white',
                         random_state=42).generate_from_frequencies(coded_dicts[t])
        ax.imshow(cloud, interpolation='bilinear')
        ax.set_title("Topic {}".format(coded_topics[t]))
        ax.axis('off')
        
    for t in range(n_valence_topics):
        ax = fig.add_subplot(spec[t+valence_gap+1, 2])
        cloud = WordCloud(background_color='white',
                         random_state=42).generate_from_frequencies(valence_dicts[t])
        ax.imshow(cloud, interpolation='bilinear')
        ax.set_title("Topic {}".format(valence_topics[t]))
        ax.axis('off')    
    # Fix up the layout - remove extra whitespace under the supertitle
    spec.tight_layout(fig, rect=[0, 0.03, 1, 0.97])
    
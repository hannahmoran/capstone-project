
import pandas as pd
import numpy as np
import gensim
import ast
import operator

from bokeh.plotting import figure, show, output_notebook
from bokeh.models import ColumnDataSource, LabelSet, Jitter, HoverTool
from bokeh.palettes import GnBu3, OrRd3
from bokeh.core.properties import value
from bokeh.transform import dodge, jitter
from bokeh.layouts import gridplot

def get_review_topics(doc, lda):
    """
    Uses gensim's get_document_topics method to get a topic 
    likelihood distribution given an LDA mode and a document. 

    :param list doc: a list of all words in the document
    :param object lda: a gensim LDA model
    :return: a dictionary where the topic numbers (0..n) are the keys and the likelihoods are the values
    :rtype: dict
    """
    bow = lda.id2word.doc2bow(doc)
    topics = lda.get_document_topics(bow, minimum_probability=0)
    topic_mix = dict(topics)
    return topic_mix


def get_sub_topic_fit(mixture):
    mixture = pd.DataFrame.from_dict(data=mixture, orient='index', columns=['fit'])
    mixture.sort_values(by='fit', inplace=True, ascending=False)
    mixture.reset_index(inplace=True)

    sub_topic_fit = mixture['fit'].iloc[1]
    
    return sub_topic_fit

def get_sub_topic(mixture):
    mixture = pd.DataFrame.from_dict(data=mixture, orient='index', columns=['fit'])
    mixture.sort_values(by='fit', inplace=True, ascending=False)
    mixture.reset_index(inplace=True)

    sub_topic = mixture['index'].iloc[1]
    
    return sub_topic

def assign_topics(df, product, final_results, load_path, input_text, encoding_type):
    """
    Assigns a topic based on a specified LDA model to each document in a set of documents.

    :param DataFrame df: a pandas DataFrame containing the docs in one column
    :param str product: string specifying the product name
    :param DataFrame final_results: pandas DataFrame containing the optimized number of topics and training runs
    :param str load_path: string specifying the folder to find the LDA model
    :param str encoding_type: string corresponding to load path that tells which type of text input was used

    The function modifies df in place, adding a column indicating the topic mixtures and the topic
    with highest likelihood for each row (doc). 
    """

    # load in the tuned LDA model for the product by getting the number of topics and training passes
    t = final_results.loc[product, 'num_topics']
    p = final_results.loc[product, 'passes']
    tn = final_results.loc[product, 'top_n removed'].astype(int)
    na = final_results.loc[product, 'n_above threshold']
    lda = gensim.models.ldamodel.LdaModel.load('../models/{}/final_models/{}_{}_{}_{}_{}'.format(load_path, 
                                                                                         product, 
                                                                                         t, p,
                                                                                         tn, na)) 

    topic_mixtures = 'Topic Mixtures {}'.format(encoding_type)
    max_topic = "{} Topic".format(encoding_type)
    max_topic_fit = "{} Fit".format(encoding_type)
    sub_topic = "{} Subtopic".format(encoding_type)
    sub_topic_fit = "{} Subtopic Fit".format(encoding_type)
    
    # Create a new column with text formatted properly
    df.loc[:,'doc'] = df[input_text].str.split(' ')

    # Use masking to select only the reviews for the product we are considering
    mask_pdct = (df['ProductId'] == product)


    # Insert a dictionary with the topic mixture likelihoods into the dataframe
    df.loc[mask_pdct, 
           topic_mixtures] = df.loc[mask_pdct, 'doc'].apply(lambda x: get_review_topics(x, lda))

    valid_1 = df[mask_pdct]
    # indicate which topic has the highest likelihood as the topic assignment
    #df.loc[mask_pdct, max_topic] = valid_1[topic_mixtures].apply(lambda x: max(ast.literal_eval(x).keys(), key=(lambda key: ast.literal_eval(x)[key])))
    df.loc[mask_pdct, max_topic] = valid_1[topic_mixtures].apply(lambda x: max(x.keys(), key=(lambda key: x[key])))


    #df.loc[mask_pdct, max_topic_fit] = valid_1[topic_mixtures].apply(lambda x: max(ast.literal_eval(x).values()))
    df.loc[mask_pdct, max_topic_fit] = valid_1[topic_mixtures].apply(lambda x: max(x.values()))

    mask_fit = ((df['ProductId']==product) & (df[max_topic_fit]<0.7))
    valid_2 = df[mask_fit]

    df.loc[mask_fit, sub_topic] = valid_2[topic_mixtures].apply(lambda x: get_sub_topic(x))

    df.loc[mask_fit, sub_topic_fit] = valid_2[topic_mixtures].apply(lambda x: get_sub_topic_fit(x))

    # Make sure the topic values are integers for max topic; cannot for subtopic b/c NaNs
    df.loc[mask_fit, max_topic] = df.loc[mask_fit, max_topic].astype(int)
    
    # We don't need this text format any longer and can drop the column
    df = df.drop('doc', axis=1, inplace=True) 
                                           
def plot_topic_distribution(df, product):
    """
    Plots a bar chart for each encoding type (vanilla, coded, valence coded) 
    to show the distribution of reviews assigned to each topic. 

    :param DataFrame df: the main dataframe containing product IDs 
                         and all review encoding sets
    :param str product: the product for which we are plotting

    :return: a bokeh gridplot object with 3 charts, one for each encoding type
    :rtype: bokeh gridplot
    """
    # get a subset of the df containing only the needed data
    data = df[df['ProductId']==product][['ProductId', 'Vanilla Topic', 
                                         'Coded Topic', "Valence Topic"]]
    
    # for each encoding type, get the distribution of topic assignments and store in an array
    vanilla = data['Vanilla Topic'].value_counts()
    vanilla_topics = ['Topic {}'.format(t) for t in vanilla.index]
    vanilla_counts = np.array(vanilla.values)
    
    coded = data['Coded Topic'].value_counts()
    coded_topics = ['Topic {}'.format(t) for t in coded.index]
    coded_counts = np.array(coded.values)
    
    valence = data['Valence Topic'].value_counts()
    valence_topics = ['Topic {}'.format(t) for t in valence.index]
    valence_counts = np.array(valence.values)
    
    # create a CDS for each of the encoding types
    source_vanilla = ColumnDataSource(data={'Vanilla Topics': vanilla_topics, 
                                            'Review Counts': vanilla_counts})
    source_coded = ColumnDataSource(data={'Coded Topics': coded_topics, 
                                          'Review Counts': coded_counts})
    source_valence = ColumnDataSource(data={'Valence Topics': valence_topics, 
                                             'Review Counts': valence_counts})
    
    # create a figure for each encoding type showing the distribution of review assignments
    #l = figure(x_range=vanilla_topics, plot_height=300, y_range=(0,550), 
    #           plot_width=325, toolbar_location=None, 
    #           title='Vanilla Topic Distribution for Product {}'.format(product))
    l = figure(x_range=vanilla_topics, plot_height=200, y_range=(0,550), 
               plot_width=225, toolbar_location=None)
    l.vbar(x='Vanilla Topics', top='Review Counts', width=0.9, source=source_vanilla)
    #l.title.text_font_size='9pt'
    l.xaxis.major_label_orientation = 0.75
    l.yaxis.axis_label = product
    
    #c = figure(x_range=coded_topics, plot_height=300, y_range=(0,550), 
    #          plot_width=325, toolbar_location=None, 
    #          title='Coded Topic Distibution for Product {}'.format(product))
    c = figure(x_range=coded_topics, plot_height=200, y_range=(0,550), 
          plot_width=225, toolbar_location=None)
    c.vbar(x='Coded Topics', top='Review Counts', width=0.9, source=source_coded)
    #c.title.text_font_size='9pt'
    c.xaxis.major_label_orientation = 0.75

    
    #r = figure(x_range=valence_topics, plot_height=300, y_range=(0,550), 
    #          plot_width=325, toolbar_location=None, 
    #          title='Valence Topic Distibution for Product {}'.format(product))
    r = figure(x_range=valence_topics, plot_height=200, y_range=(0,550), 
          plot_width=225, toolbar_location=None)
    r.vbar(x='Valence Topics', top="Review Counts", width=0.9, source=source_valence)
    #r.title.text_font_size='9pt'
    r.xaxis.major_label_orientation = 0.75

    # return the three charts for inclusion in a gridplot
    grid = [l,c,r]
    return grid
    
import download_libraries #to download all the required libraries


from selenium import webdriver
from bs4 import BeautifulSoup
from time import sleep

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import unicodedata
import re

import spacy
from spacy import displacy

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import conll2000
from nltk.chunk.util import tree2conlltags, conlltags2tree
from nltk.tag import UnigramTagger, BigramTagger
from nltk.chunk import ChunkParserI

from contractions import CONTRACTION_MAP

from textblob import TextBlob



'''
Uncomment this code if running 1st time
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('conll2000')

'''


#import import_libraries
import os



def runall(domain):
    #function for building dataset
    def build_dataset():
      url='https://inshorts.com/en/read/'+domain
      path=str(os.getcwd())+'\chromedriver.exe'
      browser=webdriver.Chrome(path)
      browser.get(url)
      count=0
      while count<2:
        browser.find_element_by_id('load-more-btn').click()
        sleep(10)
        count+=1 
      news_data = []
      news_category = url.split('/')[-1]
      data = browser.page_source
      browser.quit()
      soup = BeautifulSoup(data, 'html.parser')

      news_articles = [{'news_headline': headline.find('span', 
                                                        attrs={"itemprop": "headline"}).string,
                        'news_article': article.find('div', 
                                                      attrs={"itemprop": "articleBody"}).string,
                        'news_category': news_category}
                        
                          for headline, article in 
                            zip(soup.find_all('div', 
                                              class_=["news-card-title news-right-box"]),
                                soup.find_all('div', 
                                              class_=["news-card-content news-right-box"]))
                      ]
      news_data.extend(news_articles)
          
      df =  pd.DataFrame(news_data)
      df = df[['news_headline', 'news_article', 'news_category']]
      return df


    #Creating data of given domain
    news_df = build_dataset()



    #Stopwords
    nlp = spacy.load('en_core_web_sm')#, parse=True, tag=True, entity=True)

    tokenizer = ToktokTokenizer()
    stopword_list = nltk.corpus.stopwords.words('english')
    stopword_list.remove('no')
    stopword_list.remove('not')


    #to remove html tags(if present) in the data obtained
    def strip_html_tags(text):
        soup = BeautifulSoup(text, "html.parser")
        stripped_text = soup.get_text()
        return stripped_text


    #to remove different accented characters
    def remove_accented_chars(text):
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return text


    #To expand contractions(eg. could'nt -> could not, I'll --> I will, etc)
    def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
        
        contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),flags=re.IGNORECASE|re.DOTALL)
        def expand_match(contraction):
            match = contraction.group(0)
            first_char = match[0]
            expanded_contraction = contraction_mapping.get(match)\
                                    if contraction_mapping.get(match)\
                                    else contraction_mapping.get(match.lower())                       
            expanded_contraction = first_char+expanded_contraction[1:]
            return expanded_contraction
            
        expanded_text = contractions_pattern.sub(expand_match, text)
        expanded_text = re.sub("'", "", expanded_text)
        return expanded_text




    #To remove any special characters
    def remove_special_characters(text, remove_digits=False):
        pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
        text = re.sub(pattern, '', text)
        return text


    #Applying stemming(to get original word i.e jump from jumping,crash from crashing)
    def simple_stemmer(text):
        ps = nltk.porter.PorterStemmer()
        text = ' '.join([ps.stem(word) for word in text.split()])
        return text



    #Applying Lemmatization(to get original word i.e jump from jumping,crash from crashing))
    #(Same as stemming but it gives logical word(present in the dictionary) as compared to
    # stemming but has more complexity)

    def lemmatize_text(text):
        text = nlp(text)
        text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
        return text


    #for removing all the stopwords(a,an,the,for,my,etc)
    def remove_stopwords(text, is_lower_case=False):
        tokens = tokenizer.tokenize(text)
        tokens = [token.strip() for token in tokens]
        if is_lower_case:
            filtered_tokens = [token for token in tokens if token not in stopword_list]
        else:
            filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
        filtered_text = ' '.join(filtered_tokens)    
        return filtered_text



    #Applying all functionalities on the obtained data
    def normalize_corpus(corpus, html_stripping=True, contraction_expansion=True,
                         accented_char_removal=True, text_lower_case=True, 
                         text_lemmatization=True, special_char_removal=True, 
                         stopword_removal=True, remove_digits=True):
        
        normalized_corpus = []
        # normalize each document in the corpus
        for doc in corpus:
            # strip HTML
            if html_stripping:
                doc = strip_html_tags(doc)
            # remove accented characters
            if accented_char_removal:
                doc = remove_accented_chars(doc)
            # expand contractions    
            if contraction_expansion:
                doc = expand_contractions(doc)
            # lowercase the text    
            if text_lower_case:
                doc = doc.lower()
            # remove extra newlines
            doc = re.sub(r'[\r|\n|\r\n]+', ' ',doc)
            # lemmatize text
            if text_lemmatization:
                doc = lemmatize_text(doc)
            # remove special characters and\or digits    
            if special_char_removal:
                # insert spaces between special characters to isolate them    
                special_char_pattern = re.compile(r'([{.(-)!}])')
                doc = special_char_pattern.sub(" \\1 ", doc)
                doc = remove_special_characters(doc, remove_digits=remove_digits)  
            # remove extra whitespace
            doc = re.sub(' +', ' ', doc)
            # remove stopwords
            if stopword_removal:
                doc = remove_stopwords(doc, is_lower_case=text_lower_case)
                
            normalized_corpus.append(doc)
            
        return normalized_corpus



    # combining headline and article text
    news_df['full_text'] = news_df["news_headline"].map(str)+ '. ' + news_df["news_article"]

    # pre-process text and store the same
    news_df['clean_text'] = normalize_corpus(news_df['full_text'])
    norm_corpus = list(news_df['clean_text'])




    #news_df.to_csv('news.csv', index=False, encoding='utf-8')


    #Applying POS to get logical meaning of the text
    corpus = normalize_corpus(news_df['full_text'], text_lower_case=False, 
                              text_lemmatization=False, special_char_removal=False)

    # demo for POS tagging for sample news headline
    sentence = str(news_df.iloc[1].news_headline)
    sentence_nlp = nlp(sentence)

    # POS tagging with Spacy 
    spacy_pos_tagged = [(word, word.tag_, word.pos_) for word in sentence_nlp]
    pd.DataFrame(spacy_pos_tagged, columns=['Word', 'POS tag', 'Tag type'])

    # POS tagging with nltk
    nltk_pos_tagged = nltk.pos_tag(sentence.split())
    pd.DataFrame(nltk_pos_tagged, columns=['Word', 'POS tag'])


    #For shallow parsing,
    data = conll2000.chunked_sents()
    train_data = data[:10900]
    test_data = data[10900:] 

    #getting (word,pos tag,tag type(IOB format is used))
    wtc = tree2conlltags(train_data[1])



    #getting tags from wtc
    def conll_tag_chunks(chunk_sents):
        tagged_sents = [tree2conlltags(tree) for tree in chunk_sents]
        return [[(t, c) for (w, t, c) in sent] for sent in tagged_sents]

    #combining tags
    def combined_tagger(train_data, taggers, backoff=None):
        for tagger in taggers:
            backoff = tagger(train_data, backoff=backoff)
        return backoff 



    #N-gram tagger(to determine nth word using history of n-1 words)
    class NGramTagChunker(ChunkParserI):
        
      def __init__(self, train_sentences, 
                   tagger_classes=[UnigramTagger, BigramTagger]):
        train_sent_tags = conll_tag_chunks(train_sentences)
        self.chunk_tagger = combined_tagger(train_sent_tags, tagger_classes)

      def parse(self, tagged_sentence):
        if not tagged_sentence: 
            return None
        pos_tags = [tag for word, tag in tagged_sentence]
        chunk_pos_tags = self.chunk_tagger.tag(pos_tags)
        chunk_tags = [chunk_tag for (pos_tag, chunk_tag) in chunk_pos_tags]
        wpc_tags = [(word, pos_tag, chunk_tag) for ((word, pos_tag), chunk_tag)
                         in zip(tagged_sentence, chunk_tags)]
        return conlltags2tree(wpc_tags)


    # train chunker model  
    ntc = NGramTagChunker(train_data)

    #to see the accuracy,iob
    print(ntc.evaluate(test_data))


    #priting the dependency diagram
    svg1=displacy.render(sentence_nlp,style="dep",page=True, 
                    options={'distance': 110,
                             'arrow_stroke': 2,
                             'arrow_width': 8})

    with open("templates/Dependency_Pattern.html","w") as f:    #DEPENDENCY PATTERN
        f.write(svg1)


    #Example showing name-entity relation
    sentence = str(news_df.iloc[1].full_text)
    sentence_nlp = nlp(sentence)


    # visualize named entities

    svg2=displacy.render(sentence_nlp, style='ent', page=True) #NAME-ENTITY PATTERN
    with open("templates/Name-entity.html","w") as f:
        f.write(svg2)


    #name-entity table for the entire data
    named_entities = []
    for sentence in corpus:
        temp_entity_name = ''
        temp_named_entity = None
        sentence = nlp(sentence)
        for word in sentence:
            term = word.text 
            tag = word.ent_type_
            if tag:
                temp_entity_name = ' '.join([temp_entity_name, term]).strip()
                temp_named_entity = (temp_entity_name, tag)
            else:
                if temp_named_entity:
                    named_entities.append(temp_named_entity)
                    temp_entity_name = ''
                    temp_named_entity = None

    entity_frame = pd.DataFrame(named_entities, 
                                columns=['Entity Name', 'Entity Type'])

    #printing top 15 entities
    top_entities = (entity_frame.groupby(by=['Entity Name', 'Entity Type'])
                               .size()
                               .sort_values(ascending=False)
                               .reset_index().rename(columns={0 : 'Frequency'}))

    #top_entities.T.iloc[:,:15]
    top_entities.T.iloc[:,:15].to_csv('top_entity.csv', index=False, encoding='utf-8') #CREATING TOP ENTITIES TABLE



    #Emotion & Sentiment Analysis using Textblob library
    sentiment_scores_tb = [round(TextBlob(article).sentiment.polarity, 3) for article in news_df['clean_text']]
    sentiment_category_tb = ['positive' if score > 0 
                                 else 'negative' if score < 0 
                                     else 'neutral' 
                                         for score in sentiment_scores_tb]


    # sentiment statistics per news category
    df = pd.DataFrame([list(news_df['news_category']), sentiment_scores_tb, sentiment_category_tb]).T
    df.columns = ['news_category', 'sentiment_score', 'sentiment_category']
    df['sentiment_score'] = df.sentiment_score.astype('float')


    df.groupby(by=['news_category']).describe()


    #sentiment analysis plot
    fc = sns.catplot(x="news_category", hue="sentiment_category", 
                        data=df, kind="count", 
                        palette={"negative": "#FE2020", 
                                 "positive": "#BADD07", 
                                 "neutral": "#68BFF5"})
    plt.savefig('static/sentiment_analysis.png')     #SENTIMENT ANALYSIS GRAPH 

    html_code="<!DOCTYPE html><html lang='en'><head><title>Sentiment Analysis</title><meta charset='UTF-8'><meta name='viewport' content='width=device-width, initial-scale=1'></head><body><img src='/static/sentiment_analysis.png'></body></html>"

    with open("templates/sentiment.html","w") as f:
        f.write(html_code)


    df3=pd.concat([news_df,df],axis=1)

    df3.sort_values("sentiment_score", axis = 0, ascending = False,
                     inplace = True, na_position ='first')

    df3=df3.T.drop_duplicates().T

    df3.to_csv('Final_result.csv', index=False, encoding='utf-8')  #CREATING FINAL RESULT

    #print("Main page done")

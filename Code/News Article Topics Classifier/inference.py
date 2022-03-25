# import packages
from typing import final
import pandas as pd
from newspaper import Article
from tqdm import tqdm

import wordninja
import re
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# define class inference
class Inference:

    # function for extracting news articles
    def scrape_article(self,url):
        try:
            article = Article(url)
            article.download()
            article.parse()
            results = article.text.encode("ascii", "ignore").decode()
        except:
            article = ''
            results = article
        return results

    # create a function to clean the text
    def fulltext_clean(self,string):
        #PREPARATION
        # step 1
        # repalce all characters with a white space except these three char -  ,  . among digits/letters; 
        # Eg. keep 2,000, 3.00, covid-19
        remove_char = re.sub(r"(?!(?<=[a-zA-Z0-9])[\,\.\-](?=[a-zA-Z0-9]))[^a-zA-Z0-9 \n]"," ", string)
        # if there are more than one white spaces between words, reduce to one
        remove_spaces = re.sub('\s+', " ", remove_char).strip()   

        # step 2
        # if a word matches this pattern or is in the list then we don't want to pass it to wordninja
        # if there is hyphen, combination of letters and digits or pure capitalized letters, don't pass
        wordninja_filter = re.compile(r"-|([A-Za-z]+\d+\w*|\d+[A-Za-z]+\w*)|^[^a-z]*$")
        # if a word is in the list, don't pass it to wordninja because it can't handle the word well
        words_pass = ['qanon', 'covid']
        
        # step 3
        # set up for lemmatize
        def get_wordnet_pos(word):
            """Map POS tag to first character lemmatize() accepts"""
            tag = nltk.pos_tag([word])[0][1][0].upper()
            tag_dict = {"J": wordnet.ADJ,
                        "N": wordnet.NOUN,
                        "V": wordnet.VERB,
                        "R": wordnet.ADV}
            return tag_dict.get(tag, wordnet.NOUN)

        lemmatizer = WordNetLemmatizer()
        # step4
        # prepare stop words
        stop_words = set(stopwords.words('english'))
        # CLEANING
        # split the string by a white space
        string_isolated = remove_spaces.split()
        # check the string word by word to detect necessary split, lemmatize and remove stop word
        words_split = ''
        for el in string_isolated:
            # step 2
            # if the word matches the pattern or is in the list, then we don't pass it to wordnijia to split
            if wordninja_filter.search(el) or el.lower() in words_pass:
                temp = el
            # all the other words will be checked and be split if necessary
            else:
                temp = ' '.join(wordninja.split(el))
            # step 3: lemmatize the word
            words_lemmatized = lemmatizer.lemmatize(temp, get_wordnet_pos(temp))
            # step 4 & step 5
            if words_lemmatized.lower() not in stop_words:
                words_split += ' ' + words_lemmatized.lower()
        words_split = words_split.strip()
        return words_split

    # using parallel processing to extract articles
    def fulltext_extracter(self,filepath):
        df = pd.read_csv(filepath)
        df = df[['uid', 'url', 'keywords', 'classifiers']]
        df = df.dropna(axis = 0, how = 'any')
        urls = df['url']
        final = []
        for u in tqdm(urls):
            result = self.scrape_article(u)
            final.append(result)
        df["full_text"] = final
        return df

    # split multiple lables to rows
    def split_rows(self,df):
        df = df[['uid','full_text','classifiers']]
        df = df[df['full_text'].notnull()].reset_index(drop=True)
        # split multiple lables to rows
        data = df.set_index(['uid', 'full_text']) \
                        .apply(lambda x: x.str.split('|').explode()).reset_index()
        return data

    # apply one hot encoding
    def one_hot_encoding(self, data):
        # apply one hot encoding
        one_hot = pd.get_dummies(data['classifiers'])
        data.drop(columns = 'classifiers', axis=1, inplace=True)
        final_data = data.join(one_hot)
        return final_data

# main function
def main():
    filepath = "scraped_fulltext.csv"
    inference = Inference()
    df = inference.fulltext_extracter(filepath)
    data = inference.split_rows(df)
    # apply the function to the whole dataset
    for i in tqdm(range(len(data))):
        string = data.iloc[i,-2]
        data.iloc[i,-2] = inference.fulltext_clean(string)
    final_data = inference.one_hot_encoding(data)
    final_data.to_csv('fulltext_cleaned.csv', index=False)

if __name__ == "__main__":
    main()
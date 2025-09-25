import os
os.chdir(r"C:\Users\hp\Downloads\14. NLP WEB SCRAPING\xml_single articles")

import xml.etree.ElementTree as ET
tree = ET.parse("769952.xml")

root = tree.getroot()

root = ET.tostring(root, encoding='utf8').decode('utf8')

import re, string, unicodedata
import nltk

from bs4 import BeautifulSoup
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer

def strip_html(text):
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text()

def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)
    
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    text = re.sub(' ','',text)
    return text

sample = denoise_text(root)

# 1. Sentence tokenization
sent_tokens = sent_tokenize(sample)
print("Sentence Tokenization:")
print(sent_tokens[:5])  

# 2. Word tokenization
word_tokens = word_tokenize(sample)
print("\nWord Tokenization:")
print(word_tokens[:20])

# 3. Lowercasing and removing punctuation
word_tokens = [word.lower() for word in word_tokens if word.isalpha()]  # remove numbers/punctuations
print("\nLowercased words without punctuation:")
print(word_tokens[:20])

# 4. Removing stopwords
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in word_tokens if word not in stop_words]
print("\nWords after removing stopwords:")
print(filtered_words[:20])

# 5. Stemming
stemmer = LancasterStemmer()
stemmed_words = [stemmer.stem(word) for word in filtered_words]
print("\nAfter Stemming:")
print(stemmed_words[:20])

# 6. Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
print("\nAfter Lemmatization:")
print(lemmatized_words[:20])

from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter

text_data = " ".join(lemmatized_words)  # Convert list of words to single string

wordcloud = WordCloud(width=800, height=400,background_color='white',max_words=100,colormap='viridis').generate(text_data)

# . Display Word Cloud
plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud of XML Text", fontsize=15)
plt.show()



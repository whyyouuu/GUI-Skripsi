import pandas as pd
import re
import numpy as np
import ast
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

class globalFunction(object):
    def data_processing(self, text):
        text = text.lower()
        text = re.sub(r"https\S+|www\S+https\S+", '',text, flags=re.MULTILINE)
        text = re.sub(r'#[A-Za-z0-9]+','',text)
        text = re.sub(r'@[A-Za-z0-9]+','',text)
        text = re.sub("'", "", text)
        text = re.sub('[()!?]', ' ', text)
        text = re.sub('\[.*?\]',' ', text)
        text = re.sub("[^a-z0-9]"," ", text)
        
        #Case Folding
        text = text.lower() # mengubah ke huruf kecil
        
        #Tokenize
        regexp = RegexpTokenizer(r'\w+|$[0-9]+|\S+')
        text = regexp.tokenize(text)
    
        #Stopword
        stopword = set(stopwords.words('english'))
        text = [token for token in text if token.lower() not in stopword]
        
        #Stemming
        stemmer = PorterStemmer()
        text = [stemmer.stem(word) for word in text]

        text = [' '.join(text)]

        return text
    

    def tts(self, text):
        dataset = pd.read_csv("data/Hasil Processing Imbalance.csv")
        dataset["processing_result"] = dataset["processing_result"].apply(lambda x: ast.literal_eval(x))
        X = dataset["processing_result"].tolist()
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(X)
        sequences = tokenizer.texts_to_sequences(X)
        max_length = max([len(s) for s in sequences])
        sequence = tokenizer.texts_to_sequences(text)
        padding = pad_sequences(sequence, maxlen=max_length)
        return padding

    def predict(self, content):
        model = load_model("TestModelFix")
        predict = model.predict(content)
        label = ["1", "0", "2"]
        result = label[np.argmax(predict)]
        return result

    

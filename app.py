# pip3 install flask
# python3 01-tutorial.py 
from flask import Flask, redirect, url_for, render_template,flash,request
import re
import io
import random
import string # to process standard python strings
import warnings
import numpy as np

import json
import pickle
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
import tensorflow as tf
import tflearn
nltk.download('popular', quiet=True) # for downloading packages

#Reading in the corpus
with open('database/data.json') as fin:
    data = json.load(fin)
    
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
lemmer = WordNetLemmatizer()
stemmer = LancasterStemmer()

def StemTokens(tokens):
    return [stemmer.stem(token.lower()) for token in tokens]
def LemTokens(tokens):
    return [lemmer.lemmatize(token.lower()) for token in tokens]
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

contractions_dict = { "ain’t": "are not", "’s":" is", "aren’t": "are not", "can’t": "cannot", "can’t’ve": "cannot have", "‘cause": "because", "could’ve": "could have", "couldn’t": "could not", "couldn’t’ve": "could not have", "didn’t": "did not", "doesn’t": "does not", "don’t": "do not", "hadn’t": "had not", "hadn’t’ve": "had not have", "hasn’t": "has not", "haven’t": "have not", "he’d": "he would", "he’d’ve": "he would have", "he’ll": "he will", "he’ll’ve": "he will have", "how’d": "how did", "how’d’y": "how do you", "how’ll": "how will", "I’d": "I would", "I’d’ve": "I would have", "I’ll": "I will", "I’ll’ve": "I will have", "I’m": "I am", "I’ve": "I have", "isn’t": "is not", "it’d": "it would", "it’d’ve": "it would have", "it’ll": "it will", "it’ll’ve": "it will have", "let’s": "let us", "ma’am": "madam", "mayn’t": "may not", "might’ve": "might have", "mightn’t": "might not", "mightn’t’ve": "might not have", "must’ve": "must have", "mustn’t": "must not", "mustn’t’ve": "must not have", "needn’t": "need not", "needn’t’ve": "need not have", "o’clock": "of the clock", "oughtn’t": "ought not", "oughtn’t’ve": "ought not have", "shan’t": "shall not", "sha’n’t": "shall not", "shan’t’ve": "shall not have", "she’d": "she would", "she’d’ve": "she would have", "she’ll": "she will", "she’ll’ve": "she will have", "should’ve": "should have", "shouldn’t": "should not", "shouldn’t’ve": "should not have", "so’ve": "so have", "that’d": "that would", "that’d’ve": "that would have", "there’d": "there would", "there’d’ve": "there would have", "they’d": "they would", "they’d’ve": "they would have","they’ll": "they will",
"they’ll’ve": "they will have", "they’re": "they are", "they’ve": "they have", "to’ve": "to have", "wasn’t": "was not", "we’d": "we would", "we’d’ve": "we would have", "we’ll": "we will", "we’ll’ve": "we will have", "we’re": "we are", "we’ve": "we have", "weren’t": "were not","what’ll": "what will", "what’ll’ve": "what will have", "what’re": "what are", "what’ve": "what have", "when’ve": "when have", "where’d": "where did", "where’ve": "where have", "i'm" : "i am","there's":"there is","i've":"i have","what's":"what is","can't":"can not","don't":"do not","i'll":"i will","we're":"we are","it's":"it is","yr":"year",
"who’ll": "who will", "who’ll’ve": "who will have", "who’ve": "who have", "why’ve": "why have", "will’ve": "will have", "won’t": "will not", "won’t’ve": "will not have", "would’ve": "would have", "wouldn’t": "would not", "wouldn’t’ve": "would not have", "y’all": "you all", "y’all’d": "you all would", "y’all’d’ve": "you all would have", "y’all’re": "you all are", "y’all’ve": "you all have", "you’d": "you would", "you’d’ve": "you would have", "you’ll": "you will", "you’ll’ve": "you will have", "you’re": "you are", "you’ve": "you have"}

contractions_re = re.compile('(%s)'%'|'.join(contractions_dict.keys()))
def expand_contractions(s, contractions_dict=contractions_dict):
  def replace(match):
    return contractions_dict[match.group(0)]
  return contractions_re.sub(replace, s)


def preprocess(sentences):
    sentences  = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",sentences).split())
    sentences  = sentences.replace('https?:\/\/.*[\r\n]*', '')
    # sentences  = sentences.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    sentences  = sentences.replace('\d+', '')
    sentences  = sentences.replace('[^\w\s]', '')
    sentences  = expand_contractions(sentences)
    return sentences
  
try:
    with open("data.pickle", "rb") as f:
        words, labels, X, y = pickle.load(f)
except:
#Tokenisation
  words = []
  sentences = []
  labels = []
  docs_x = []
  docs_y = []
  bert_inputs = []

  for intent in data['intents']:
        for pattern in intent['patterns']:
              pattern = preprocess(pattern)
              tokenize = LemNormalize(pattern)
              words.extend(tokenize)
              sentences.append(pattern)
              docs_x.append(tokenize)
              docs_y.append(intent["tag"])
        labels.append(intent["tag"])

  # Preprocessing
  words = LemTokens(words)
  words = sorted(list(set(words)))

  labels = sorted(labels)

  X = []
  y = []

  init = [0]*(len(labels))

  for x, doc in enumerate(docs_x):
        bag = []
        find_words = LemTokens(doc)
        
        for w in words:
              if w in find_words:
                    bag.append(1)
              else:
                    bag.append(0)
                    
        output_row = init[:]
        output_row[labels.index(docs_y[x])] = 1
        
        X.append(bag)
        y.append(output_row)
  with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, X, y), f)

def model(n_layers=2,n_neurons=8,input_shape=[None, len(X[0])]):
      
      net = tflearn.input_data(shape=input_shape)
      for _ in range(n_layers):
        net = tflearn.fully_connected(net, n_neurons)
      net = tflearn.fully_connected(net, len(y[0]), activation="softmax")
      net = tflearn.regression(net)
      return tflearn.DNN(net)

try:
  model = model()
  model.load('model.tflearn')
except:
  tf.compat.v1.reset_default_graph()
  model = model()
  model.fit(X, y, n_epoch=800, batch_size=8, show_metric=True)
  model.save("model.tflearn")


def bag_of_words(s, words):
    bag = [0]*len(words)
    s_words = LemNormalize(s)
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return np.array([bag])
# Generating response
def response(user_response):

    # sent_tokens.append(user_response)
    # TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    # tfidf = TfidfVec.fit_transform(sent_tokens)
    # vals = cosine_similarity(tfidf[-1], tfidf)
    # idx=vals.argsort()[0][-2]
    # flat = vals.flatten()
    # flat.sort()
    # req_tfidf = flat[-2]
    # if(req_tfidf==0):
    #     robo_response=robo_response+"Can you please elaborate?"
    #     return robo_response
    # else:
    #     robo_response = robo_response+sent_tokens[idx]
    #     return robo_response
    results = model.predict(bag_of_words(user_response, words))
    results_index = np.argmax(results)
    tag = labels[results_index]
    for tg in data["intents"]:
        if tg['tag'] == tag:
            return random.choice(tg['responses'])
    return "Sorry i don't understand can you please rephrase";

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
 
chat = ["Hello, I'm Robo your personal chatbot!.\n Can I help you with anything?"];
# STATE = 'greeting'
# pending_state = 'get_info'
@app.route("/",methods=["POST","GET"]) # set the route
def home(): # return html
  if request.method == "POST":
    user_respond = request.form["user_respond"]
    chat.append(user_respond)

    chat.append(response(preprocess(user_respond.lower())))
    # else:
      # STATE=pending_state
  return render_template("chatview.html", respond=chat)



if __name__ == "__main__":
  app.run(debug = True) # debug = True, updates the changes
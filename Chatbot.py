import nltk
#nltk.download('popular')
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
import json
import pickle
import numpy as np
from keras.models import load_model
import random

#data_file = open("data.json").read()
data_file = open('Fullset.json', encoding='utf-8').read()
intents = json.loads(data_file)
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('model.h5')


def get_input():
    userText = str(input("Enter input"))
    return chatbot_response(userText)


def chatbot_response(m):
    ints = brain(m, model)
    response = get_botchat(ints, intents)
    return response


def get_botchat(ints, intents):
    try:
        tag = ints[0]['intent']
        list_of_intents = intents['intents']
        for i in list_of_intents:
            if i['tags'] == tag:
                result = random.choice(i['responses'])
                # eval(i['functions'])
                break
    except IndexError:
        result = "Sorry, I didn't understand that."

    print(result)
    return result


def get_botchat(ints, intents):
    try:
        tag = ints[0]['intent']
        list_of_intents = intents['intents']
        for i in list_of_intents:
            if i['tags'] == tag:
                result = random.choice(i['responses'])
                # eval(i['functions'])
                break
    except IndexError:
        result = "Please ask Complete Question."

    print(result)
    return result


def brain(sen, mod):
    prediction = bow(sen, words, show_details=False)
    res = model.predict(np.array([prediction]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # resilts=[[greeting,0.01],[bye,0.02],[thank,0.9]]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
        # return_list=[{intent:greeting,probability:0.9}]
    return return_list


def bow(sen, words, show_details=True):
    sentence_words = clean_sentense(sen)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return (np.array(bag))


def clean_sentense(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


while True:
    get_input()
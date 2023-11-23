import nltk
#nltk.download('popular')
#nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()

import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

words = []
classes = []
documents = []
ignore_words = ['!', '?']
#data_file = open('data.json').read()
# Open and read the JSON file with 'utf-8' encoding
data_file = open('Fullset.json', encoding='utf-8').read()

intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word = nltk.word_tokenize(pattern)
        words.extend(word)
        documents.append((word, intent['tags']))
        if intent['tags'] not in classes:
            classes.append(intent['tags'])

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_words]

words = sorted(list(set(words)))
classes = sorted(list(set(classes)))
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))
training = []
output = [0] * len(classes)
for document in documents:
    bag = []
    pattern_words = document[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    print(pattern_words)
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)
    output_row = list(output)
    output_row[classes.index(document[1])] = 1
    print(bag)
    training.append([bag, output_row])
    print(training)
random.shuffle(training)
training = np.asarray(training, dtype ='object')
train_x = list(training[:,0])
train_y = list(training[:,1])

model =Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation="softmax"))
sgd = SGD(learning_rate=0.01, momentum = 0.9,nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy',metrics=['accuracy'])
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose =1)
model.save('model.h5',hist)

print("Model Saved")
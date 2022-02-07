# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 23:00:57 2022

@author: avita
"""

import json 
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import colorama 
colorama.init()
from colorama import Fore, Style, Back

import random
import pickle

with open("intents.json") as file:
    data = json.load(file)


def chat():
    # load trained model
    model = keras.models.load_model('chat_model')

    # load tokenizer object
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # load label encoder object
    with open('label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)

    # parameters
    max_len = 20
    
    while True:
        print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL, end="")
        inp = input()
        if inp.lower() == "quit":
            break

        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                             truncating='post', maxlen=max_len))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])

        for i in data['intents']:
            if i['tag'] == tag:
                return(Fore.GREEN + "ChatBot:" + Style.RESET_ALL , np.random.choice(i['responses']))

        # print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL,random.choice(responses))

print(Fore.YELLOW + "Start messaging with the bot (type quit to stop)!" + Style.RESET_ALL)
chat()

def main():
    st.sidebar.title("NLP Bot")
    st.title(""" Initialize the bot by clicking the "Initialize bot" button.""")
    bot = st.text_input("Start messaging with the bot (type quit to stop)!")
    results=''
    if st.button("Click Here"):
        results= chat()
    st.success(results)

if __name__=='__main__':
    main()

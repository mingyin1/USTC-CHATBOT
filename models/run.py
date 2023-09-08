
import numpy as np
import pickle
import operator
from keras.preprocessing import sequence
from keras.layers import Embedding
from keras.layers import Input, Dense, LSTM, TimeDistributed, Bidirectional, Dropout, Concatenate, RepeatVector, Activation, Dot
from keras.layers import concatenate, dot                    
from keras.models import Model
#from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.initializers import TruncatedNormal
#import pydot
import os, re
from keras.preprocessing import sequence
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import jieba
import requests
def act_weather(city):
    url = 'http://wthrcdn.etouch.cn/weather_mini?city=' + city
    page = requests.get(url)
    data = page.json()
    temperature = data['data']['wendu']
    notice = data['data']['ganmao']
    outstrs = "地点： %s\n气温： %s\n注意： %s" % (city, temperature, notice)
    return outstrs + ' EOS'




class Run:
    def __init__(self):
        current_path = os.path.dirname(__file__) + "/"
        self.question = np.load(current_path + 'pad_question.npy',allow_pickle=True)
        self.answer = np.load(current_path + 'pad_answer.npy',allow_pickle=True)
        self.answer_o = np.load(current_path + 'answer_o.npy',allow_pickle=True)
        with open(current_path + 'vocab_bag.pkl', 'rb') as f:
            self.words = pickle.load(f)
        with open(current_path + 'pad_word_to_index.pkl', 'rb') as f:
            self.word_to_index = pickle.load(f)
        with open(current_path + 'pad_index_to_word.pkl', 'rb') as f:
            self.index_to_word = pickle.load(f)
        vocab_size = len(self.word_to_index) + 1
        self.maxLen=20
        truncatednormal = TruncatedNormal(mean=0.0, stddev=0.05)
        embed_layer = Embedding(input_dim=vocab_size, 
								output_dim=100, 
								mask_zero=True,
								input_length=None,
								embeddings_initializer= truncatednormal)
        LSTM_encoder = LSTM(512,
							return_sequences=True,
							return_state=True,
							kernel_initializer= 'lecun_uniform',
							name='encoder_lstm'
								)
        LSTM_decoder = LSTM(512, 
							return_sequences=True, 
							return_state=True, 
							kernel_initializer= 'lecun_uniform',
							name='decoder_lstm'
						)

		#encoder输入 与 decoder输入
        input_question = Input(shape=(None, ), dtype='int32', name='input_question')
        input_answer = Input(shape=(None, ), dtype='int32', name='input_answer')

        input_question_embed = embed_layer(input_question)
        input_answer_embed = embed_layer(input_answer)


        encoder_lstm, question_h, question_c = LSTM_encoder(input_question_embed)

        decoder_lstm, _, _ = LSTM_decoder(input_answer_embed, 
										initial_state=[question_h, question_c])

        attention = dot([decoder_lstm, encoder_lstm], axes=[2, 2])
        attention = Activation('softmax')(attention)
        context = dot([attention, encoder_lstm], axes=[2,1])
        decoder_combined_context = concatenate([context, decoder_lstm])


		# Has another weight + tanh layer as described in equation (5) of the paper
        decoder_dense1 = TimeDistributed(Dense(256,activation="tanh"))
        decoder_dense2 = TimeDistributed(Dense(vocab_size,activation="softmax"))
        output = decoder_dense1(decoder_combined_context) # equation (5) of the paper
        output = decoder_dense2(output) # equation (6) of the paper

        self.model = Model([input_question, input_answer], output)

        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

        self.model.load_weights(current_path + 'W--200-0.2621-.h5')
        self.model.summary()

        self.question_model = Model(input_question, [encoder_lstm, question_h, question_c])
        self.question_model.summary()
        answer_h = Input(shape=(512,))
        answer_c = Input(shape=(512,))
        encoder_lstm = Input(shape=(self.maxLen,512))
        target, h, c = LSTM_decoder(input_answer_embed, initial_state=[answer_h, answer_c])
        attention = dot([target, encoder_lstm], axes=[2, 2])
        attention_ = Activation('softmax')(attention)
        context = dot([attention_, encoder_lstm], axes=[2,1])
        decoder_combined_context = concatenate([context, target])
        output = decoder_dense1(decoder_combined_context) # equation (5) of the paper
        output = decoder_dense2(output) # equation (6) of the paper
        self.answer_model = Model([input_answer, answer_h, answer_c, encoder_lstm], [output, h, c, attention_])
        self.answer_model.summary()
    def input_question(self, seq):
        seq = jieba.lcut(seq.strip(), cut_all=False)
        sentence = seq
        try:
            seq = np.array([self.word_to_index[w] for w in seq])
        except KeyError:
            seq = np.array([36874, 165, 14625])
        seq = sequence.pad_sequences([seq], maxlen=self.maxLen,
                                            padding='post', truncating='post')
        print(seq)
        return seq, sentence

    def decode_greedy(self, seq, sentence):
        question = seq
        for index in question[0]:
            if int(index) == 5900:
                for index_ in question[0]:
                    if index_ in [7851, 11842,2406, 3485, 823, 12773, 8078]:
                        return act_weather(self.index_to_word[index_])
        answer = np.zeros((1, 1))
        attention_plot = np.zeros((20, 20))
        answer[0, 0] = self.word_to_index['BOS']
        i=1
        answer_ = []
        flag = 0
        encoder_lstm_, question_h, question_c = self.question_model.predict(x=question, verbose=1)
    #     print(question_h, '\n')
        while flag != 1:
            prediction, prediction_h, prediction_c, attention = self.answer_model.predict([
                answer, question_h, question_c, encoder_lstm_
            ])
            attention_weights = attention.reshape(-1, )
            attention_plot[i] = attention_weights
            word_arg = np.argmax(prediction[0, -1, :])#
            answer_.append(self.index_to_word[word_arg])
            if word_arg == self.word_to_index['EOS']  or i > 20:
                flag = 1
            answer = np.zeros((1, 1))
            answer[0, 0] = word_arg
            question_h = prediction_h
            question_c = prediction_c
            i += 1
        result = ' '.join(answer_)
        return result
    def predict(self, question):
        seq ,sentence = self.input_question(question)
        temp_answer0 = self.decode_greedy(seq, sentence)
        temp_answer1 = temp_answer0.replace(' ','')
        temp_answer2 = temp_answer1.replace('E','')
        temp_answer3 = temp_answer2.replace('S','')
        final_answer = temp_answer3.replace('O','')
        return final_answer

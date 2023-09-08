import sys,os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from hyperparameter import HyperParameters as hp
from transformers import TFBertForSequenceClassification,BertTokenizer
import tensorflow as tf
from untils import encode_examples,split_dataset, convert_example_to_feature
import numpy as np
from random import randint

class NETWORK(object) :
    def __init__(self, load_path = "model/classifier229+9925.h5"):
        self.querytimes = [0] * hp.num_classes
        # model initialization
        self.model = TFBertForSequenceClassification.from_pretrained(hp.bert_path, num_labels=hp.num_classes)

        # optimizer Adam recommended
        optimizer = tf.keras.optimizers.Adam(learning_rate=hp.learning_rate,epsilon=1e-08, clipnorm=1)
        # we do not have one-hot vectors, we can use sparce categorical cross entropy and accuracy
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
        self.model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

        if load_path != None:
            self.model.load_weights(hp.current_path + load_path)        
        self.model.summary()

    def train(self, train_data):
        # split data
        #train_data, val_data = split_dataset(df_raw)
        # train dataset
        ds_train_encoded = encode_examples(train_data).shuffle(10000).batch(hp.batch_size)
        # val dataset
        #ds_val_encoded = encode_examples(val_data).batch(hp.batch_size)
        # fit model
        bert_history = self.model.fit(ds_train_encoded, epochs=hp.number_of_epochs)
        loss = int(bert_history.history['loss'][-1] * 10000)
        acc = int(bert_history.history['accuracy'][-1] * 10000)
        model_path = "model/classifier" + str(loss) + "+" + str(acc) + ".h5"
        self.model.save_weights(hp.current_path + model_path)
        return model_path

    def predict(self, input):
        model_input = convert_example_to_feature(input)
        output = self.model.predict([model_input['input_ids']]).logits[0]
        print(output)
        Id = np.argmax(output[1:]) + 1
        print(Id)
        print(output[Id])
        if output[Id] < 1:
            return None
        else :
            Ans = hp.Ansers[Id][self.querytimes[Id] if self.querytimes[Id] < len(hp.Ansers[Id]) else randint(0, len(hp.Ansers[Id]) - 1)]
            self.querytimes[Id] = self.querytimes[Id] + 1
            return Ans


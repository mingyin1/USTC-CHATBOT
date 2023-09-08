from transformers import TFBertForSequenceClassification,BertTokenizer
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
import pandas as pd
from hyperparameter import HyperParameters as hp
# tokenizer
tokenizer = BertTokenizer.from_pretrained(hp.bert_path)
def convert_example_to_feature(review):
  # combine step for tokenization, WordPiece vector mapping, adding special tokens as well as truncating reviews longer than the max length
	return tokenizer.encode_plus(review, 
	            add_special_tokens = True, # add [CLS], [SEP]
	            max_length = hp.max_length, # max length of the text that can go to BERT
	            pad_to_max_length = True, # add [PAD] tokens
	            return_attention_mask = True, # add attention mask to not focus on pad tokens
		    truncation=True
	          )

# map to the expected input to TFBertForSequenceClassification, see here 
def map_example_to_dict(input_ids, attention_masks, token_type_ids, label):
    return {
      "input_ids": input_ids,
      "token_type_ids": token_type_ids,
      "attention_mask": attention_masks,
  }, label


def encode_examples(ds, limit=-1):
    # prepare list, so that we can build up final TensorFlow dataset from slices.
    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []
    label_list = []
    if (limit > 0):
        ds = ds.take(limit)
  
    for index, row in ds.iterrows():
        review = row["text"]
        label = row["y"]
        bert_input = convert_example_to_feature(review)
  
        input_ids_list.append(bert_input['input_ids'])
        token_type_ids_list.append(bert_input['token_type_ids'])
        attention_mask_list.append(bert_input['attention_mask'])
        label_list.append([label])
    return tf.data.Dataset.from_tensor_slices((input_ids_list, attention_mask_list, token_type_ids_list, label_list)).map(map_example_to_dict)

def split_dataset(df):
    train_set, val_set = train_test_split(df, 
        stratify=df['label'],
        test_size=0.2, 
        random_state=42)

    return train_set, val_set

def read_data():
	# read data
    df_raw = pd.read_csv(hp.current_path + hp.data_path,sep="|",header=None,names=["text","label"])    
	# transfer label
    df_label = pd.DataFrame({"label":hp.labels ,"y":list(range(hp.num_classes))})
    train_data = pd.merge(df_raw,df_label,on="label",how="left")
    for x,y in train_data.iterrows():
        print(x,y)
    print(train_data)
    return train_data

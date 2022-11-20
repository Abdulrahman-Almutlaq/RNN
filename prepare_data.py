import helper
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import dvc.api
import os
import pickle

def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    # TODO: Implement Function
    # getting all words without repetition
    words = set(text)
    
    # declaring the tuple
    vocab_to_int = {}
    int_to_vocab = {}
    
    for integer,word in enumerate(words):
        vocab_to_int[word] = integer
        int_to_vocab[integer] = word
        
    # return tuple
    return (vocab_to_int, int_to_vocab)

def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenized dictionary where the key is the punctuation and the value is the token
    """
    # TODO: Implement Function
    punc_to_tok = {".":"||Period||", ",":"||Comma||", '"':"||QuotationMark||",
                   ";":"||Semicolon||", "!":"||ExclamationMark||",
                "?":"||QuestionMark||", "(":"||LeftParentheses||",
               ")":"||RightParentheses||", "-":"||Dash||", ".":"||Period||", "\n":"||Return||"}
    
    return punc_to_tok

def batch_data(words, sequence_length, batch_size):
     # get number of targets we can make
    n_targets = len(words) - sequence_length
    # initialize feature and target 
    feature, target = [], []
    
    # loop through all targets we can make
    for i in range(n_targets):
        x = words[i : i+sequence_length]    # get some words from the given list
        y = words[i+sequence_length]        # get the next word to be the target
        feature.append(x)
        target.append(y)
    
    feature_tensor, target_tensor = torch.from_numpy(np.array(feature)), torch.from_numpy(np.array(target))
    # create data
    data = TensorDataset(feature_tensor, target_tensor)
    # create dataloader
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
     # return a dataloader
    return dataloader

data_dir = './data/Seinfeld_Scripts.txt'
text = helper.load_data(data_dir)
helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)
int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()





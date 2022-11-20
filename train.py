import torch.nn as nn
import torch
import numpy as np
import sys
from mlem.api import save
from prepare_data import vocab_to_int, batch_data, int_text
import yaml
import json




class RNN(nn.Module):
    
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5):
        """
        Initialize the PyTorch RNN Module
        :param vocab_size: The number of input dimensions of the neural network (the size of the vocabulary)
        :param output_size: The number of output dimensions of the neural network
        :param embedding_dim: The size of embeddings, should you choose to use them        
        :param hidden_dim: The size of the hidden layer outputs
        :param dropout: dropout to add in between LSTM/GRU layers
        """
        super(RNN, self).__init__()
        # TODO: Implement function
        
        # set class variables
        self.out_size = output_size
        self.n_layers = n_layers
        self.hid_dim = hidden_dim
        self.vocab_size = vocab_size
        # define model layers
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                            dropout = dropout, batch_first=True)
        
        self.dropout = nn.Dropout(dropout)
        
        self.fc = nn.Linear(hidden_dim, output_size)
    
    def forward(self, nn_input, hidden):
        """
        Forward propagation of the neural network
        :param nn_input: The input to the neural network
        :param hidden: The hidden state        
        :return: Two Tensors, the output of the neural network and the latest hidden state
        """
        # TODO: Implement function   
        batch_size = nn_input.size(0)

        reduced_inputs = self.embed(nn_input.long())
        
        lstm_out, hidden = self.lstm(reduced_inputs,hidden)
        
        output = lstm_out.contiguous().view(-1, self.hid_dim)
        
        output = self.fc(output)
        
        output = output.view(batch_size, -1, self.out_size)
        
        output = output[:, -1]
        
        # return one batch of output word scores and the hidden state
        return output, hidden
    
    
    def init_hidden(self, batch_size):
        '''
        Initialize the hidden state of an LSTM/GRU
        :param batch_size: The batch_size of the hidden state
        :return: hidden state of dims (n_layers, batch_size, hidden_dim)
        '''
        # Implement function
        
        # initialize hidden state with zero weights, and move to GPU if available
        
        weights = next(self.parameters()).data
        
        gpu_availability = torch.cuda.is_available()
        
        if (gpu_availability):
            hidden = (weights.new(self.n_layers, batch_size, self.hid_dim).zero_().cuda(),
                  weights.new(self.n_layers, batch_size, self.hid_dim).zero_().cuda())
        else:
            hidden = (weights.new(self.n_layers, batch_size, self.hid_dim).zero_(),
                      weights.new(self.n_layers, batch_size, self.hid_dim).zero_())

        return hidden

def forward_back_prop(rnn, optimizer, criterion, inp, target, hidden):
    """
    Forward and backward propagation on the neural network
    :param decoder: The PyTorch Module that holds the neural network
    :param decoder_optimizer: The PyTorch optimizer for the neural network
    :param criterion: The PyTorch loss function
    :param inp: A batch of input to the neural network
    :param target: The target output for the batch of input
    :return: The loss and the latest hidden state Tensor
    """
    
    # TODO: Implement Function
    
    # move data to GPU, if available
    
    inp, target = inp.cuda(), target.cuda()
    criterion = criterion.cuda()
    # perform backpropagation and optimization
    
    # so we don't go so many steps in the back in order to back-propagate
    hidden = tuple([each.data for each in hidden])
    
    optimizer.zero_grad()
    
    output, hidden = rnn(inp,hidden)
    
    loss = criterion(output.squeeze(),target.long())
    loss.backward()
    
    nn.utils.clip_grad_norm_(rnn.parameters(), 5)
    optimizer.step()
        
    # return the loss over a batch and the hidden state produced by our model
    return loss.item(), hidden

def train_rnn(rnn, batch_size, optimizer, criterion, n_epochs, show_every_n_batches=100):
    batch_losses = []
    all_avg_loss = []

    rnn.train()
    try:            
        print("Training for %d epoch(s)..." % n_epochs)
        for epoch_i in range(1, n_epochs + 1):

            # initialize hidden state
            hidden = rnn.init_hidden(batch_size)

            for batch_i, (inputs, labels) in enumerate(train_loader, 1):
                
                
                # make sure you iterate over completely full batches, only
                n_batches = len(train_loader.dataset)//batch_size
                if(batch_i > n_batches):
                    break

                # forward, back prop
                loss, hidden = forward_back_prop(rnn, optimizer, criterion, inputs, labels, hidden)          
                # record loss
                batch_losses.append(loss)
                all_avg_loss.append(loss)


                # printing loss stats
                if batch_i % show_every_n_batches == 0:
                    print('Epoch: {:>4}/{:<4}  Loss: {}\n'.format(
                        epoch_i, n_epochs, np.average(batch_losses)))
                    batch_losses = []
                    
    except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory, retrying batch',sys.stdout)
                    sys.stdout.flush()
                    for p in rnn.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    loss, hidden = forward_back_prop(rnn, optimizer, criterion, inputs, labels, hidden)          
                    # record loss
                    batch_losses.append(loss)

                    # printing loss stats
                    if batch_i % show_every_n_batches == 0:
                        print('Epoch: {:>4}/{:<4}  Loss: {}\n'.format(
                            epoch_i, n_epochs, np.average(batch_losses)))
                        batch_losses = []
                else:
                    raise e
    dictionary = {
        "avg_training_loss": np.average(all_avg_loss)
    }

    with open("metrics.json", "w") as outfile:
        json.dump(dictionary, outfile)
    # returns a trained rnn
    return rnn



# Data params
# Sequence Length
sequence_length = 10 # of words in a sequence
# Batch Size
batch_size = 512

# data loader - do not change
train_loader = batch_data(int_text, sequence_length, batch_size)

# Training parameters
# Number of Epochs
num_epochs = 1
# Learning Rate
learning_rate = 0.01

# Model parameters
# Vocab size

vocab_size = len(vocab_to_int)
# Output size 111434272603048975244
output_size = len(vocab_to_int)
# Embedding Dimension
embedding_dim = 30
# Hidden Dimension
hidden_dim = 256
# Number of RNN Layers
n_layers = 1

# Show stats for every n number of batches
show_every_n_batches = 500
torch.cuda.empty_cache()


# create model and move to gpu if available
rnn = RNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5)
# if train_on_gpu:
rnn.cuda()

# defining loss and optimization functions for training
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# training the model
trained_rnn = train_rnn(rnn, batch_size, optimizer, criterion, num_epochs, show_every_n_batches)

d = {
    "Base":{
    'Data_path':'data',
    'Seed':10,
    'Data_source': 'Udacity'
    },
    "Train":{
    'Num_epochs':num_epochs,
    'Learning_rate':learning_rate,
    'Vocab_size': len(vocab_to_int),
    'Output_size':len(vocab_to_int),
    'Embedding_dim':embedding_dim,
    'hidden_dim': hidden_dim,
    'Num_layers': n_layers,
    'Batch_size':batch_size,
    'Sequence_length':sequence_length,
    
    },
    "Other_params":{
    'Optimizer':"Adam",
    'Learning_rate':'CrossEntropyLoss',    
    }
    }

with open('params.yaml', 'w') as yaml_file:
    yaml.dump(d, yaml_file)


for batch_i, (inputs, labels) in enumerate(train_loader, 1):
    inputs = inputs.cuda()
    hidden = rnn.init_hidden(batch_size)
    save(
        obj = trained_rnn,
        path= "models/RNN"
    )
    break
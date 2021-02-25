"""
Basic RNN baseline from Hube2019

"""

import sys

sys.path.append('../')

print(sys.path)

import torch
from torch import nn, optim
from torchtext import data, vocab
from utils import text_processing
import numpy as np


def get_vocab_and_iterators(data_tsv_path, glove_path, device, train_test_val_split=[0.7,0.2,0.1]):
    """
    This uses torchtext to build a vocabulary (includes the embedding matrix) and train/val/test iterators
    Documentation: https://pytorch.org/text/data.html#
    Tutorial: http://anie.me/On-Torchtext/
    """
    LABEL = data.Field(sequential=False, use_vocab=False)
    # Preprocessing is run after tokenization
    TEXT = data.Field(preprocessing=None, tokenize=text_processing.tokenize, lower=True)

    dataset = data.TabularDataset(data_tsv_path,format='TSV',fields=[('label',LABEL),('text',TEXT)])

    # Splitnumbers are [train, test, val], but returns [train, val, test]
    trainset, valset, testset = dataset.split(train_test_val_split, stratified=True, strata_field='label')

    glove_vectors = vocab.Vectors(glove_path,unk_init=None) # Initialize OOVs as zeros

    TEXT.build_vocab(trainset, vectors=glove_vectors)

    ### Iterators
    train_iter, val_iter, test_iter = data.BucketIterator.splits((trainset, valset, testset), batch_sizes=(100,100,100), sort_key=lambda x: len(x.text), shuffle=True, device=device)

    return TEXT.vocab, train_iter, val_iter, test_iter


class RNN(nn.Module):
    def __init__(self, hidden_size, embed_weight_matrix):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_weight_matrix.shape[1]
        # Freezes weights
        self.embed_layer = nn.Embedding.from_pretrained(embed_weight_matrix, freeze=True)
        # Hidden output: 1 x batch_size x hidden_size
        self.gru = nn.GRU(self.embed_size, hidden_size)
        # Output size is 1
        self.linear = nn.Linear(hidden_size,1)

    # Inputs x are indices of tokens
    def forward(self, x):
        # x.shape: [seq_len, batch_size]
        embedded = self.embed_layer(x)
        # embedded.shape: [seq_len, batch_size, embed_dim]
        gru_output, hidden = self.gru(embedded)
        # gru_output.shape: [seq_len, batch_size, embed_dim]
        # Only keep last output
        out = gru_output[-1,:,:].squeeze()
        # out.shape: [batch_size, embed_dim]
        out = self.linear(out).squeeze()
        # out.shape: [batch_size]        
        return out

def train(model, train_iter, val_iter, epochs, model_save_path, model_load_path, device):

    model.to(device)
    
    model = torch.load(model_load_path)
    for param in model.parameters():
        param.requires_grad = False
    model.linear.weight.requires_grad = True
    model.linear.bias.requires_grad = True
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())
    sigmoid_fun = nn.Sigmoid() # For measuring accuracy

    for epoch in range(epochs):
        running_train_loss = 0.0
        running_train_acc = 0.0
        model.train() # Activates dropout and batch normalization
        for i, batch in enumerate(train_iter):
            # print(batch)
            output = model(batch.text)
            # print(output,'###',batch.label.type_as(output))
            loss = criterion(output, batch.label.type_as(output))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

            # Accuracy
            output_sigmoid = sigmoid_fun(output)
            acc_tensor = (output_sigmoid > 0.5) == batch.label
            acc_tensor = acc_tensor.type(torch.DoubleTensor)
            accuracy = sum(acc_tensor)/len(acc_tensor)
            running_train_acc += accuracy
                    
        # Validation
        running_val_loss = 0.0
        running_val_acc = 0.0
        with torch.no_grad():
            model.eval() # Deactivates dropout and batch normal
            for i, batch in enumerate(val_iter):
                output = model(batch.text)
                loss = criterion(output, batch.label.type_as(output))
                running_val_loss += loss.item()

                # Accuracy
                output_sigmoid = sigmoid_fun(output)
                acc_tensor = (output_sigmoid > 0.5) == batch.label
                acc_tensor = acc_tensor.type(torch.DoubleTensor)
                accuracy = sum(acc_tensor)/len(acc_tensor)
                running_val_acc += accuracy

        print('Epoch {:2d}, Train loss: {:.4f}, Train acc: {:.4f} | Val loss: {:.4f}, Val acc: {:.4f}'.format(epoch,running_train_loss/len(train_iter),running_train_acc/len(train_iter),running_val_loss/len(val_iter),running_val_acc/len(val_iter)))

    torch.save(model, model_save_path)

    
def load_and_test_model(model_save_path,test_iter):
    criterion = nn.BCEWithLogitsLoss()
    sigmoid_fun = nn.Sigmoid()

    model = torch.load(model_save_path)
    model.eval()

    running_loss = 0.0
    running_acc = 0.0
    running_precision = 0.0
    running_recall = 0.0
    running_f1score = 0.0

    for i, batch in enumerate(test_iter):
        output = model(batch.text)
        loss = criterion(output, batch.label.type_as(output))

        running_loss += loss.item()

        # Accuracy
        output_sigmoid = sigmoid_fun(output)
        predict=(output_sigmoid > 0.5).type(torch.uint8)
        target=batch.label
        acc_tensor = predict == target
        acc_tensor = acc_tensor.type(torch.DoubleTensor)
        accuracy = sum(acc_tensor)/len(acc_tensor)

        running_acc += accuracy

        #F1-Score
        tp = (predict*target)
        fp = (predict*(1-target))
        fn = ((1-predict)*target)
        recall = torch.sum(tp,dtype=torch.float64)/torch.sum(tp+fn)
        precision = torch.sum(tp,dtype=torch.float64)/torch.sum(tp+fn)
        running_precision += precision
        running_recall += recall
        
        # running_f1score += (2/((1/precision)+(1/recall)))
        # running_f1score += 2*((precision*recall)/(precision+recall))


    print('Test loss: {:.4f}, Test acc: {:.4f}'.format(running_loss/len(test_iter),running_acc/len(test_iter)))
    print('Test loss: {:.4f}, Test recall: {:.4f}'.format(running_loss/len(test_iter),running_precision/len(test_iter)))
    print('Test loss: {:.4f}, Test precision: {:.4f}'.format(running_loss/len(test_iter),running_recall/len(test_iter)))


if __name__ == "__main__":
    if len(sys.argv) == 4:
        # Specify the paths with arguments
        tsv_path = sys.argv[1] 
        glove_path = sys.argv[2]
        model_save_path = sys.argv[3]
        model_load_patch = sys.argv[4]
    else:
        # Use default paths
        tsv_path = '../../data/processed/neutral_and_biased.tsv'
        glove_path = '../../data/other/glove.6B.100d.txt'
        model_save_path = '../../model_checkpoints/solution.ckpt'
        model_load_path= '../../model_checkpoints/rnn_Pryzant.ckpt'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #torch.device('cpu')
    print('Cuda available: {}'.format(torch.cuda.is_available()))

    vocabulary, train_iter, val_iter, test_iter = get_vocab_and_iterators(tsv_path,glove_path,device=device)
    
    print('Batches in Train: {}, Val: {}, Test: {}'.format(len(train_iter),len(val_iter),len(test_iter)))
    
    print('Embedding matrix shape: {}'.format(vocabulary.vectors.shape))
    
    rnn = RNN(100,vocabulary.vectors)
    
    print('Total parameters: {}'.format(sum(p.numel() for p in rnn.parameters())))
    print('Trainable parameters: {}'.format(sum(p.numel() for p in rnn.parameters() if p.requires_grad)))
    
    train(rnn, train_iter, val_iter, 10, model_save_path, model_load_path, device)

    print('Testing')
    load_and_test_model(model_save_path, test_iter)

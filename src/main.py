import sys

sys.path.append('../')
import load_data
import os
import time
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe
from torchtext import data, vocab
import numpy as np
from torchsummary import summary
from sklearn.model_selection import StratifiedKFold

from utils import text_processing
from models.RNN import RNN
from models.RCNN import RCNN
from models.selfAttention import SelfAttention



def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)
      

def train_model(model, optim, train_iter, batch_size, epoch):
    total_epoch_loss = 0.0
    total_epoch_acc = 0.0
    if torch.cuda.is_available():
        model.cuda()

    loss_fn = F.cross_entropy
    #optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    steps = 0
    model.train()
    for idx, batch in enumerate(train_iter):
        text = batch.text
        target = batch.label
        target = torch.autograd.Variable(target).long()
        if torch.cuda.is_available():
            text = text.cuda()
            target = target.cuda()
        # if( (text.shape[0],text.shape[1]) != (100,1)):
            # print('Validation - Input Shape Issue:',text.shape)
            # continue
        if (text.size()[0] is not batch_size):# One of the batch returned by BucketIterator has length different than 100.
            continue
        optim.zero_grad()
        prediction = model(text) # Shape [100,2]?
        loss = loss_fn(prediction, target)

        pred_labels = torch.max(prediction, 1)[1].view(target.size()).data

        
        correct_preds = (pred_labels == target).type(torch.DoubleTensor)
        # Accuracy
        acc = sum(correct_preds)/len(correct_preds)

        loss.backward()
        clip_gradient(model, 1e-1)
        optim.step()
        steps += 1

        #if steps % 100 == 0:
        #    print (f'Epoch: {epoch+1}, Idx: {idx+1}, Training Loss: {loss.item():.4f}, Training Accuracy: {acc.item(): .2f}%')
        
        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()
        
    return total_epoch_loss/len(train_iter), total_epoch_acc/len(train_iter)


def eval_model(model, val_iter, batch_size):
    total_epoch_loss = 0.0
    total_epoch_acc = 0.0
    total_epoch_precision = 0.0
    total_epoch_recall = 0.0
    loss_fn = F.cross_entropy

    num_batches_processed = 0
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(val_iter):
            text = batch.text
            #text.size()=[batch_size,seq_leng]
            # if( (text.shape[0],text.shape[1]) != (100,1)):
                # print('Validation - Input Shape Issue:',text.shape)
                # continue
            if (text.size()[0] is not batch_size): #One of the batch returned by BucketIterator has length different than 100.
                continue
            num_batches_processed += 1 # Can be different from len(val_iter) because of line above

            target = batch.label
            target = torch.autograd.Variable(target).long()
            if torch.cuda.is_available():
                text = text.cuda()
                target = target.cuda()
            prediction = model(text)
            loss = loss_fn(prediction, target)


            pred_labels = torch.max(prediction, 1)[1].view(target.size()).data
            correct_preds = (pred_labels == target).type(torch.DoubleTensor)
            # Accuracy
            acc = sum(correct_preds)/len(correct_preds)

            # Precision & Recall
            true_positives = pred_labels*target

            num_predicted = torch.sum(pred_labels, dtype=torch.float64)
            if num_predicted.item() == 0.0:
                precision = 1.0
            else:
                precision = (torch.sum(true_positives, dtype=torch.float64)/num_predicted).item()

            num_target = torch.sum(target, dtype=torch.float64)
            if num_target.item() == 0.0:
                recall = 1.0
            else:
                recall = (torch.sum(true_positives, dtype=torch.float64)/num_target).item()

            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()
            total_epoch_precision += precision
            total_epoch_recall += recall

    return total_epoch_loss/num_batches_processed, total_epoch_acc/num_batches_processed, total_epoch_precision/num_batches_processed, total_epoch_recall/num_batches_processed 


def get_text_field(data_tsv_path, glove_path, max_embeddings):
    """
    Generates a text field (including vocab/word embeddings) 
    For PRYZANT data (or any tsv with id, label, text columns)
    """
    LABEL = data.Field(sequential=False, use_vocab=False, batch_first=True)
    # Preprocessing is run after tokenization
    TEXT = data.Field(preprocessing=None, tokenize=text_processing.tokenize, lower=True, batch_first=True)
    # sequential=True, tokenize=tokenize, lower=True, include_lengths=True,fix_length=200
    
    dataset = data.TabularDataset(data_tsv_path,format='TSV',fields=[('label',LABEL),('text',TEXT)])
    # Splitnumbers are [train, test, val], but returns [train, val, test]
    #trainset, valset, testset = dataset.split(train_test_val_split, stratified=True, strata_field='label')
    glove_vectors = vocab.Vectors(glove_path,unk_init=None) # Initialize OOVs as zeros
    TEXT.build_vocab(dataset, vectors=glove_vectors, max_size=max_embeddings)

    return TEXT



def cross_validation(model, batch_size=20, n_folds=5, num_epochs=20):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    tsv_filepath = "../data/processed/hube_balanced.tsv"
    glove_path = "../data/other/glove.6B.100d.txt"
    pryzant_path = "../data/processed/pryzant2019_full.tsv" # For generating text field

    layers = [module for module in model.modules()]
    max_embeddings = layers[1].num_embeddings # Embedding size

    # Generate text field with vocab
    # vocab is max. as big as the embedding layer of the pretrained model
    TEXT = get_text_field(pryzant_path, glove_path, max_embeddings)
    print("Generated text field")

    LABEL = data.Field(sequential=False, use_vocab=False, batch_first=True)
    label_field = ('label', LABEL)
    text_field = ('text', TEXT)

    examples = []
    labels = []
    with open(tsv_filepath) as infile:
        for line in infile:
            splits = line.split('\t')
            # First column is label, second column is text
            example = data.Example.fromlist([splits[0],splits[1]], [label_field,text_field])
            examples.append(example)
            labels.append(int(splits[0]))
        
    examples = np.array(examples)
    
    # Save model weights
    model_save_path = "../model_checkpoints/temp.ckpt"
    torch.save({
                'model_state_dict': model.state_dict()
                }, model_save_path)

    acc_sum = 0.0
    prec_sum = 0.0
    rec_sum = 0.0
    skf = StratifiedKFold(n_splits=n_folds)
    print('Starting cross validation with {} folds'.format(n_folds))
    print("Accuracy \t Precision \t Recall")
    for train_idx, test_idx in skf.split(np.zeros(len(labels)), labels):
        trainset = data.Dataset(examples[train_idx], fields=[label_field,text_field])
        testset = data.Dataset(examples[test_idx], fields=[label_field,text_field])
    
        train_iter = data.BucketIterator(trainset, batch_size=batch_size, train=True, sort_key=lambda x: len(x.text),repeat=False, shuffle=True, device=device)
        test_iter = data.BucketIterator(testset, batch_size=batch_size, train=False, sort_key=lambda x: len(x.text),repeat=False, shuffle=True, device=device)

        # Load model weights for each fold
        checkpoint = torch.load(model_save_path)
        for param in model.parameters():
            param.requires_grad = False
            model.label.weight.requires_grad = True
            model.label.bias.requires_grad = True
        
        model.load_state_dict(checkpoint['model_state_dict'])

        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
        
        for epoch in range(num_epochs):
            _,_ = train_model(model, optim, train_iter, batch_size, epoch)
            
        loss, acc, prec, rec = eval_model(model,test_iter, batch_size)
        print(str(acc) +'\t'+ str(prec) +'\t'+ str(rec))

        acc_sum += acc 
        prec_sum += prec 
        rec_sum += rec 
    
    acc_mean = acc_sum/n_folds
    prec_mean = prec_sum/n_folds
    rec_mean = rec_sum/n_folds

    f1_score = 2*(prec_mean*rec_mean)/(prec_mean+rec_mean)

    print("Averages: Accuracy: {} \t Precision: {} \t Recall: {} \t F1-Score: {}".format(acc_mean,prec_mean,rec_mean,f1_score))
    return acc_mean, prec_mean, rec_mean, f1_score



if __name__ == "__main__":


    ### EITHER LOAD PRETRAINED MODEL
    CHECKPOINT_DIR = "../model_checkpoints/"
    # Remove the "map_location" part if running on GPU
    selfattn_model = torch.load(CHECKPOINT_DIR + 'PryzantFull.cktp')

    #### OR CREATE NEW (NOT PRETRAINED) MODEL
    PRYZANT_PATH = "../data/processed/hube_balanced.tsv"
    GLOVE_PATH = "../data/other/glove.6B.100d.txt"
    text_field = get_text_field(PRYZANT_PATH, GLOVE_PATH, None)
    embeddings = text_field.vocab.vectors
    selfattn_model = RCNN(20, 2, 100, embeddings.shape[0], embeddings.shape[1], embeddings)
    
    ### RUN CROSS VALIDATION
    cross_validation(selfattn_model, batch_size=20, num_epochs=20)

"""
victor_rnnkolja = torch.load(CHECKPOINT_DIR + 'Victor_rnnkolja.ckpt', map_location=torch.device('cpu'))
victor_rnnkolja

ckpt_selfattn # Complete model
"""
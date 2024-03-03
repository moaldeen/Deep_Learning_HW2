########## Mohammed aldeen ##############
import torch.optim as optim
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random
import os
from scipy.special import expit
import random
import sys
import json
import re
from torch.utils.data import DataLoader, Dataset
import pickle
import models



########## DATA HANDLING AND PREPARATION ##########

class DataPreparer(Dataset):
    def __init__(self, annotation_file, directory, vocabulary, index_map):
        self.annotation_file = annotation_file
        self.directory = directory
        self.audio_visual_content = load_audio_data(annotation_file)
        self.index_map = index_map
        self.vocabulary = vocabulary
        self.data_pairs = link_annotations(directory, vocabulary, index_map)

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, index):
        assert (index < self.__len__())
        audio_name, text = self.data_pairs[index]
        content = torch.Tensor(self.audio_visual_content[audio_name])
        content += torch.Tensor(content.size()).random_(0, 2000)/10000
        return torch.Tensor(content), torch.Tensor(text)

class TestDataLoader(Dataset):
    def __init__(self, data_path):
        self.audio_content = []
        files = os.listdir(data_path)
        for file in files:
            identifier = file.split('.npy')[0]
            content = np.load(os.path.join(data_path, file))
            self.audio_content.append((identifier, content))

    def __len__(self):
        return len(self.audio_content)

    def __getitem__(self, index):
        return self.audio_content[index]

def build_vocabulary(min_word_frequency):
    with open('training_label.json', 'r') as file:
        annotations = json.load(file)

    word_frequency = {}
    for item in annotations:
        for sentence in item['caption']:
            processed_sentence = re.sub('[.!,;?]]', ' ', sentence).split()
            for word in processed_sentence:
                word = word.rstrip('.')  # Remove trailing period
                word_frequency[word] = word_frequency.get(word, 0) + 1

    vocabulary = {}
    for word, freq in word_frequency.items():
        if freq > min_word_frequency:
            vocabulary[word] = freq

    token_mapping = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
    index_to_word = {index + len(token_mapping): word for index, word in enumerate(vocabulary)}
    word_to_index = {word: index + len(token_mapping) for index, word in enumerate(vocabulary)}
    index_to_word.update({value: key for key, value in token_mapping.items()})
    word_to_index.update({key: value for key, value in token_mapping.items()})

    return index_to_word, word_to_index, vocabulary

def split_and_tokenize_sentence(sentence, vocabulary, word_to_index):
    sentence = re.sub(r'[.!,;?]', ' ', sentence).split()
    tokenized_sentence = [word_to_index.get(word, 3) for word in sentence]  # 3 is for <UNK>
    tokenized_sentence = [1] + tokenized_sentence + [2]  # Add <SOS> and <EOS>
    return tokenized_sentence

def link_annotations(annotation_file, vocabulary, word_to_index):
    linked_data = []
    with open(annotation_file, 'r') as file:
        annotations = json.load(file)
    for item in annotations:
        for sentence in item['caption']:
            processed_sentence = split_and_tokenize_sentence(sentence, vocabulary, word_to_index)
            linked_data.append((item['id'], processed_sentence))
    return linked_data

def load_audio_data(directory):
    audio_data_map = {}
    files = os.listdir(directory)
    for file in files:
        content = np.load(os.path.join(directory, file))
        audio_data_map[file.split('.npy')[0]] = content
    return audio_data_map


def train_network(model, current_epoch, loader, loss_function):
    model.train()
    print(current_epoch)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for batch_index, (audio_features, labels, sequence_lengths) in enumerate(loader):
        audio_features, labels = audio_features.cuda(), labels.cuda()
        audio_features, labels = Variable(audio_features), Variable(labels)

        optimizer.zero_grad()
        seq_log_probs, seq_predictions = model(audio_features, target_sentences=labels, mode='train', training_steps=current_epoch)

        labels = labels[:, 1:]  
        loss = compute_loss(seq_log_probs, labels, sequence_lengths, loss_function)
        loss.backward()
        optimizer.step()

    print(f'Epoch: {current_epoch}, Loss: {loss.item():.3f}')

def evaluate_network(loader, model):
    model.eval()
    for batch_index, (audio_features, labels, sequence_lengths) in enumerate(loader):
        audio_features, labels = audio_features.cuda(), labels.cuda()
        audio_features, labels = Variable(audio_features), Variable(labels)

        seq_log_probs, seq_predictions = model(audio_features, mode='inference')
        return seq_predictions[:3], labels[:3]  

def minibatch_collate(batch_data):
    batch_data.sort(key=lambda x: len(x[1]), reverse=True)
    audio_data, text_sequences = zip(*batch_data)
    audio_tensor = torch.stack(audio_data, 0)

    sequence_lengths = [len(sequence) for sequence in text_sequences]
    max_length = max(sequence_lengths)
    padded_sequences = torch.zeros(len(text_sequences), max_length).long()
    for i, sequence in enumerate(text_sequences):
        end = sequence_lengths[i]
        padded_sequences[i, :end] = torch.LongTensor(sequence[:end])

    return audio_tensor, padded_sequences, sequence_lengths




def main():
    annotation_file = 'training_label.json'
    training_data_dir = 'training_data/feat'
    index_to_word, word_to_index, vocabulary = build_vocabulary(4)
    train_dataset = DataPreparer(annotation_file, training_data_dir, vocabulary, word_to_index)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=5, shuffle=True, num_workers=8, collate_fn=minibatch_collate)
    
    testing_data_dir = 'testing_data/feat'
    testing_annotation_file = 'testing_label.json'
    test_dataset = DataPreparer(testing_annotation_file, testing_data_dir, vocabulary, word_to_index)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True, num_workers=8, collate_fn=minibatch_collate)
   
    epochs_n = 100
    model_save_location = 'model'
    
    if not os.path.exists(model_save_location):
        os.makedirs(model_save_location)
    
    with open(os.path.join(model_save_location, 'i2wData.pickle'), 'wb') as file:
         pickle.dump(index_to_word, file)
    
    vocab_size = len(index_to_word) + 4 
    loss_fn = nn.CrossEntropyLoss()
    encoder = models.EncoderNet()  
    decoder = models.DecoderNet(512, vocab_size, vocab_size, 1024, 0.3) 
    model = models.ModelMain(encoder=encoder, decoder=decoder) 
    
    start_time = time.time()
    for epoch in range(epochs_n):
        train_network(model, epoch + 1, train_dataloader, loss_fn)
        evaluate_network(test_dataloader, model)
    
    end_time = time.time()
    torch.save(model.state_dict(), os.path.join(model_save_location, 'model0.pth'))
    print(f"Training completed in {(end_time - start_time):.3f} seconds.")

main()



#!/usr/bin/env python
# coding: utf-8


#####################################################################################
#############################   DEEP LEARNING HOMEWORK 2 ############################
#############################   Mohammed Aldeen          ############################
#####################################################################################

import sys
import torch
import json
from torch.utils.data import DataLoader
from bleu_eval import BLEU
import Main

import pickle

if not torch.cuda.is_available():
    modelIP = torch.load('model/model0.h5', map_location=lambda storage, loc: storage)
else:
    modelIP = torch.load('model/model0.h5')



files_dir = 'testing_data/feat'
i2w, w2i, dictionary = Main.dictonaryFunc(word_min=5)

test_dataset = Main.test_dataloader(files_dir)
test_dataloader = Main.DataLoader(dataset=test_dataset, batch_size=1, shuffle=True, num_workers=8)

model = modelIP.cuda()

ss = Main.test(test_dataloader, model, i2w)

with open('test_output.txt', 'w') as f:
    for id, s in ss:
        f.write('{},{}\n'.format(id, s))



# Bleu Eval
test = json.load(open('testing_label.json','r'))
output = 'test_output.txt'
result = {}

with open(output,'r') as f:
    for line in f:
        line = line.rstrip()
        comma = line.index(',')
        test_id = line[:comma]
        caption = line[comma+1:]
        result[test_id] = caption

        bleu=[]
for item in test:
    score_per_video = []
    captions = [x.rstrip('.') for x in item['caption']]
    score_per_video.append(BLEU(result[item['id']],captions,True))
    bleu.append(score_per_video[0])
average = sum(bleu) / len(bleu)
print("Average bleu score is " + str(average))

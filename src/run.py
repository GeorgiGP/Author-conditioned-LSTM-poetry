#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2024/2025
#############################################################################
###
### Домашно задание 3
###
#############################################################################

import sys
import torch

import utils
import generator
import train
import model
import pickle


from parameters import *

startChar = '{'
endChar = '}'
unkChar = '@'
padChar = '|'

if len(sys.argv)>1 and sys.argv[1] == 'prepare':
    testCorpus, trainCorpus, char2id, auth2id =  utils.prepareData(corpusFileName, startChar, endChar, unkChar, padChar)
    pickle.dump(testCorpus, open(testDataFileName, 'wb'))
    pickle.dump(trainCorpus, open(trainDataFileName, 'wb'))
    pickle.dump(char2id, open(char2idFileName, 'wb'))
    pickle.dump(auth2id, open(auth2idFileName, 'wb'))
    print('Data prepared.')

if len(sys.argv)>1 and sys.argv[1] == 'train':
    testCorpus = pickle.load(open(testDataFileName, 'rb'))
    trainCorpus = pickle.load(open(trainDataFileName, 'rb'))
    char2id = pickle.load(open(char2idFileName, 'rb'))
    auth2id = pickle.load(open(auth2idFileName, 'rb'))

    lm = model.LSTMLanguageModelPack(char_emb_size, hid_size, auth2id, char2id, unkChar, padChar, endChar, lstm_layers=lstm_layers, dropout=dropout).to(device)
    if len(sys.argv)>2: lm.load(sys.argv[2], device)

    optimizer = torch.optim.Adam(lm.parameters(), lr=learning_rate)
    train.trainModel(trainCorpus, lm, optimizer, epochs, batchSize)
    lm.save(modelFileName)
    print('Model perplexity: ',train.perplexity(lm, testCorpus, batchSize))

if len(sys.argv)>1 and sys.argv[1] == 'perplexity':
    testCorpus = pickle.load(open(testDataFileName, 'rb'))
    char2id = pickle.load(open(char2idFileName, 'rb'))
    auth2id = pickle.load(open(auth2idFileName, 'rb'))
    lm = model.LSTMLanguageModelPack(char_emb_size, hid_size, auth2id, char2id, unkChar, padChar, endChar, lstm_layers=lstm_layers, dropout=dropout).to(device)
    lm.load(modelFileName,device)
    print('Model perplexity: ',train.perplexity(lm, testCorpus, batchSize))

if len(sys.argv)>1 and sys.argv[1] == 'generate':
    if len(sys.argv)>2: auth = sys.argv[2]
    else:
        print('Usage: python run.py generate author [seed [temperature]]')
    if len(sys.argv)>3: seed = sys.argv[3]
    else: seed = startChar

    assert seed[0] == startChar

    if len(sys.argv)>4: temperature = float(sys.argv[4])
    else: temperature = defaultTemperature
 
    char2id = pickle.load(open(char2idFileName, 'rb'))
    auth2id = pickle.load(open(auth2idFileName, 'rb'))
    lm = model.LSTMLanguageModelPack(char_emb_size, hid_size, auth2id, char2id, unkChar, padChar, endChar, lstm_layers=lstm_layers, dropout=dropout).to(device)
    lm.load(modelFileName,device)
    
    authid = auth2id.get(auth,0)
    if authid==0: print('Авторът не е известен.')
    print(generator.generateText(lm, char2id, auth, seed, temperature=temperature))



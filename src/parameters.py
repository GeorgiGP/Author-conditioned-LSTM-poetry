import torch

corpusFileName = 'corpusPoems'
modelFileName = 'modelLSTM'
trainDataFileName = 'trainData'
testDataFileName = 'testData'
char2idFileName = 'char2id'
auth2idFileName = 'auth2id'

#device = torch.device("cuda:0")
#device = torch.device("cpu")
device = torch.device("mps")

batchSize = 64
char_emb_size = 128

hid_size = 256
lstm_layers = 2
dropout = 0.2

epochs = 10
learning_rate = 0.003

defaultTemperature = 0.4

#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2024/2025
#############################################################################
###
### Домашно задание 3
###
#############################################################################

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

#################################################################
####  LSTM с пакетиране на партида
#################################################################

class LSTMLanguageModelPack(torch.nn.Module):
    def preparePaddedBatch(self, source):
        device = next(self.parameters()).device
        m = max(len(s) for (a,s) in source)
        sents = [[self.word2ind.get(w,self.unkTokenIdx) for w in s] for (a,s) in source]
        auths = [self.auth2id.get(a,0) for (a,s) in source]
        sents_padded = [ s+(m-len(s))*[self.padTokenIdx] for s in sents]
        return torch.t(torch.tensor(sents_padded, dtype=torch.long, device=device)), torch.tensor(auths, dtype=torch.long, device=device)
    
    def save(self,fileName):
        torch.save(self.state_dict(), fileName)
    
    def load(self,fileName,device):
        self.load_state_dict(torch.load(fileName,device))

    def __init__(self, embed_size, hidden_size, auth2id, word2ind, unkToken, padToken, endToken, lstm_layers, dropout):
        super(LSTMLanguageModelPack, self).__init__()
        #############################################################################
        ###  Тук следва да се имплементира инициализацията на обекта
        ###  За целта може да копирате съответния метод от програмата за упр. 13
        ###  като направите добавки за повече слоеве на РНН, влагане за автора и dropout
        #############################################################################
        #### Начало на Вашия код.
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.auth2id = auth2id
        self.word2ind = word2ind
        self.unkTokenIdx = word2ind[unkToken]
        self.padTokenIdx = word2ind[padToken]
        self.endTokenIdx = word2ind[endToken]
        self.lstm_layers = lstm_layers
        
        self.char_embed = nn.Embedding(len(word2ind), embed_size)
        self.auth_embed = nn.Embedding(len(auth2id), hidden_size * lstm_layers * 2)
        self.lstm = nn.LSTM(input_size=embed_size, 
                            hidden_size=hidden_size, 
                            num_layers=lstm_layers, 
                            dropout=dropout)
        self.dropout_layer = nn.Dropout(dropout)
        self.projection = nn.Linear(hidden_size, len(word2ind))
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.padTokenIdx)
            
        #### Край на Вашия код
        #############################################################################

    def forward(self, source):
        #############################################################################
        ###  Тук следва да се имплементира forward метода на обекта
        ###  За целта може да копирате съответния метод от програмата за упр. 13
        ###  като направите добавка за dropout и началните скрити вектори
        ######################zz#######################################################
        #### Начало на Вашия код.
        
        X, A = self.preparePaddedBatch(source)
        
        X_in = X[:-1]
        X_target = X[1:]
        batch_size = A.shape[0]
        auth_vecs = self.auth_embed(A)
        auth_vecs = auth_vecs.view(batch_size, self.lstm_layers, 2, self.hidden_size).permute(2, 1, 0, 3).contiguous()
        
        h0 = auth_vecs[0]
        c0 = auth_vecs[1]
        
        x_embedded = self.char_embed(X_in)
        
        lengths = (X_in != self.padTokenIdx).sum(dim=0).cpu()
        packed_input = pack_padded_sequence(x_embedded, lengths, enforce_sorted=False)
        
        packed_output, _ = self.lstm(packed_input, (h0, c0))
        output, _ = pad_packed_sequence(packed_output)
        
        output = self.dropout_layer(output)
        logits = self.projection(output)
        loss = self.loss_fn(logits.reshape(-1, len(self.word2ind)), X_target.reshape(-1))
        
        return loss
    
        #### Край на Вашия код
        #############################################################################


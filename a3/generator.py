#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2024/2025
#############################################################################
###
### Домашно задание 3
###
#############################################################################

import numpy as np
import torch
import torch.nn.functional as F

def generateText(model, char2id, auth, startSentence, limit=1000, temperature=1.):
    # model е инстанция на обучен LSTMLanguageModelPack обект
    # char2id е речник за символите, връщащ съответните индекси
    # startSentence е началния низ стартиращ със символа за начало '{'
    # limit е горна граница за дължината на поемата
    # temperature е температурата за промяна на разпределението за следващ символ
    
    result = startSentence[1:]

    #############################################################################
    ###  Тук следва да се имплементира генерацията на текста
    #############################################################################
    #### Начало на Вашия код.
    
    model.eval()
    device = next(model.parameters()).device
    id2char = {v: k for k, v in char2id.items()}
    
    with torch.no_grad():
        auth_idx = model.auth2id.get(auth, 0)
        auth_tensor = torch.tensor([auth_idx], dtype=torch.long, device=device)
        
        auth_vecs = model.auth_embed(auth_tensor)
        auth_vecs = auth_vecs.view(1, model.lstm_layers, 2, model.hidden_size).permute(2, 1, 0, 3).contiguous()
        h, c = auth_vecs[0], auth_vecs[1]
        
        input_ids = [char2id.get(char, model.unkTokenIdx) for char in startSentence]
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(1)
        
        embedded = model.char_embed(input_tensor)
        output, (h, c) = model.lstm(embedded, (h, c))
        
        last_char_idx = input_ids[-1]
        
        for _ in range(limit):
            inp = torch.tensor([[last_char_idx]], dtype=torch.long, device=device)
            emb = model.char_embed(inp)
            
            out, (h, c) = model.lstm(emb, (h, c))
            logits = model.projection(out.squeeze(0).squeeze(0))
            
            if temperature == 0:
                next_char_idx = torch.argmax(logits).item()
            else:
                logits = logits / temperature
                probs = F.softmax(logits, dim=0)
                next_char_idx = torch.multinomial(probs, 1).item()
            
            if next_char_idx == model.endTokenIdx:
                break
            
            next_char = id2char.get(next_char_idx, '@')
            result += next_char
            last_char_idx = next_char_idx
	
    #### Край на Вашия код
    #############################################################################

    return result

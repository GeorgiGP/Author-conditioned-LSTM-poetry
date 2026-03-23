import torch
import sys

import model
import generator

EMBED_SIZE = 16
HIDDEN_SIZE = 32
LSTM_LAYERS = 2
DROPOUT = 0.1
BATCH_SIZE = 2

START_CHAR = '{'
END_CHAR = '}'
UNK_CHAR = '@'
PAD_CHAR = '|'

def run_tests():
    print("=== Стартиране на тестове за валидация ===")
    
    print("[1/4] Създаване на тестови речници...")
    chars = [START_CHAR, END_CHAR, UNK_CHAR, PAD_CHAR, 'а', 'б', 'в', 'г', ' ', '\n']
    char2id = {c: i for i, c in enumerate(chars)}
    authors = ['Иван Вазов', 'Христо Ботев']
    auth2id = {a: i for i, a in enumerate(authors)}
    
    print("      -> Речниците са създадени успешно.")

    print("[2/4] Инициализация на модела (model.py)...")
    device = torch.device("cpu")
    
    try:
        lm = model.LSTMLanguageModelPack(
            embed_size=EMBED_SIZE,
            hidden_size=HIDDEN_SIZE,
            auth2id=auth2id,
            word2ind=char2id,
            unkToken=UNK_CHAR,
            padToken=PAD_CHAR,
            endToken=END_CHAR,
            lstm_layers=LSTM_LAYERS,
            dropout=DROPOUT
        ).to(device)
        print("      -> Моделът е инициализиран успешно.")
    except Exception as e:
        print(f"      !!! ГРЕШКА при инициализация: {e}")
        return

    print("[3/4] Тестване на 'forward' метод (тренировъчен режим)...")
    try:
        dummy_batch = [
            ('Иван Вазов', '{абаг}'),
            ('Христо Ботев', '{вв}')
        ]
        
        loss = lm(dummy_batch)
        
        if isinstance(loss, torch.Tensor) and loss.item() > 0:
            print(f"      -> Forward pass премина успешно. Loss: {loss.item():.4f}")
        else:
            print("      !!! ГРЕШКА: Forward pass не върна валидна загуба.")
            return
    except Exception as e:
        print(f"      !!! ГРЕШКА по време на forward pass: {e}")
        import traceback
        traceback.print_exc()
        return

    print("[4/4] Тестване на генератора (generator.py)...")
    try:
        seed_text = "{аб"
        target_author = "Иван Вазов"
        
        generated_text = generator.generateText(
            model=lm, 
            char2id=char2id, 
            auth=target_author, 
            startSentence=seed_text, 
            limit=20, 
            temperature=1.0
        )
        
        print(f"      -> Генериран текст: '{generated_text}'")
        
        if isinstance(generated_text, str) and len(generated_text) > 0:
            print("      -> Генераторът работи успешно.")
        else:
            print("      !!! ГРЕШКА: Генераторът върна празен или невалиден резултат.")
    except Exception as e:
        print(f"      !!! ГРЕШКА при генерация: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n=== ВСИЧКИ ТЕСТОВЕ ПРЕМИНАХА УСПЕШНО! ===")
    print("Вашата имплементация изглежда структурно валидна.")

if __name__ == "__main__":
    run_tests()
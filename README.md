# LSTM Language Model for Bulgarian Poetry Generation

## Author-Conditioned Character-Level Neural Language Model

**Author:** Georgi H. Lazov
**Student ID:** 0MI0600299
**Course:** Information Retrieval and Deep Learning
**Instructor:** Prof. Stoyan Mihov
**Semester:** Winter 2025/2026

---

## Abstract

This work presents an implementation of a character-level LSTM language model for generating Bulgarian poetry conditioned on specific authors. The model uses packed sequences for efficient batch processing and incorporates author embeddings to initialize the LSTM hidden states, allowing the model to generate text in the style of different poets. The implementation includes training procedures, perplexity evaluation, and text generation with temperature-controlled sampling.

**Keywords:** LSTM, Language Model, Text Generation, Poetry, Character-Level Model, Author Conditioning, Bulgarian Language

---

## 1. Introduction

### 1.1 Motivation

Neural language models have demonstrated remarkable capabilities in generating coherent and stylistically consistent text. Character-level models are particularly well-suited for morphologically rich languages like Bulgarian and for creative text generation where capturing fine-grained patterns is essential.

### 1.2 Project Objectives

1. Implement a multi-layer LSTM language model with packed sequence processing
2. Incorporate author conditioning through learned author embeddings
3. Implement training with cross-entropy loss and perplexity evaluation
4. Develop a text generation function with temperature-controlled sampling
5. Train the model on a corpus of Bulgarian poetry

### 1.3 Corpus Used

The model was trained on the **Corpus of Bulgarian Poems** containing works from multiple Bulgarian poets:
- **Format:** Character-level sequences with author labels
- **Language:** Bulgarian
- **Content:** Poetry from various Bulgarian authors

---

## 2. Theoretical Background

### 2.1 Character-Level Language Models

Character-level language models predict the next character given a sequence of previous characters. For a sequence $c_1, c_2, ..., c_T$, the model learns:

$$P(c_t | c_1, ..., c_{t-1})$$

The total probability of a sequence is:

$$P(c_1, ..., c_T) = \prod_{t=1}^{T} P(c_t | c_1, ..., c_{t-1})$$

### 2.2 LSTM Architecture

Long Short-Term Memory (LSTM) networks address the vanishing gradient problem through gating mechanisms:

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
$$h_t = o_t \odot \tanh(C_t)$$

where:
- $f_t$ is the forget gate
- $i_t$ is the input gate
- $o_t$ is the output gate
- $C_t$ is the cell state
- $h_t$ is the hidden state

### 2.3 Author Conditioning

The key innovation in this implementation is the use of author embeddings to initialize the LSTM hidden states. For each author $a$, a learned embedding vector is transformed to initialize both the hidden state $h_0$ and cell state $c_0$:

$$[h_0, c_0] = \text{reshape}(\text{AuthorEmbed}(a))$$

This allows the model to generate text conditioned on a specific author's style.

### 2.4 Perplexity

Perplexity measures the model's predictive performance:

$$\text{PPL} = \exp\left(-\frac{1}{N}\sum_{i=1}^{N} \log P(c_i | c_1, ..., c_{i-1})\right)$$

Lower perplexity indicates better predictive performance.

---

## 3. Implementation

### 3.1 Model Architecture (`model.py`)

#### 3.1.1 LSTMLanguageModelPack

The model class implements a multi-layer LSTM with author conditioning:

```python
class LSTMLanguageModelPack(torch.nn.Module):
    def __init__(self, embed_size, hidden_size, auth2id, word2ind,
                 unkToken, padToken, endToken, lstm_layers, dropout):

        self.char_embed = nn.Embedding(len(word2ind), embed_size)
        self.auth_embed = nn.Embedding(len(auth2id), hidden_size * lstm_layers * 2)
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=lstm_layers,
                            dropout=dropout)
        self.dropout_layer = nn.Dropout(dropout)
        self.projection = nn.Linear(hidden_size, len(word2ind))
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.padTokenIdx)
```

#### 3.1.2 Packed Sequence Processing

The model uses PyTorch's `pack_padded_sequence` for efficient batch processing with variable-length sequences:

```python
def forward(self, source):
    X, A = self.preparePaddedBatch(source)

    # Author embeddings initialize hidden states
    auth_vecs = self.auth_embed(A)
    auth_vecs = auth_vecs.view(batch_size, self.lstm_layers, 2,
                                self.hidden_size).permute(2, 1, 0, 3)
    h0, c0 = auth_vecs[0], auth_vecs[1]

    # Pack sequences for efficient processing
    packed_input = pack_padded_sequence(x_embedded, lengths, enforce_sorted=False)
    packed_output, _ = self.lstm(packed_input, (h0, c0))
    output, _ = pad_packed_sequence(packed_output)

    # Apply dropout and compute loss
    output = self.dropout_layer(output)
    logits = self.projection(output)
    loss = self.loss_fn(logits.reshape(-1, len(self.word2ind)), X_target.reshape(-1))

    return loss
```

### 3.2 Text Generation (`generator.py`)

#### 3.2.1 Temperature-Controlled Sampling

The generation function implements autoregressive text generation with temperature scaling:

```python
def generateText(model, char2id, auth, startSentence, limit=1000, temperature=1.):
    # Initialize with author embedding
    auth_vecs = model.auth_embed(auth_tensor)
    h, c = auth_vecs[0], auth_vecs[1]

    # Autoregressive generation
    for _ in range(limit):
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

        result += id2char.get(next_char_idx, '@')

    return result
```

**Temperature Effects:**
- $T \to 0$: Greedy decoding (deterministic)
- $T = 1$: Standard sampling
- $T > 1$: More diverse/random outputs

### 3.3 Training (`train.py`)

```python
def trainModel(trainCorpus, lm, optimizer, epochs, batchSize):
    for epoch in range(epochs):
        np.random.shuffle(idx)
        for b in range(0, len(idx), batchSize):
            batch = [trainCorpus[i] for i in idx[b:min(b+batchSize, len(idx))]]
            H = lm(batch)
            optimizer.zero_grad()
            H.backward()
            optimizer.step()
```

---

## 4. Model Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| char_emb_size | 128 | Character embedding dimension |
| hid_size | 256 | LSTM hidden state dimension |
| lstm_layers | 2 | Number of LSTM layers |
| dropout | 0.2 | Dropout rate |
| batchSize | 64 | Training batch size |
| epochs | 10 | Number of training epochs |
| learning_rate | 0.003 | Adam optimizer learning rate |
| defaultTemperature | 0.4 | Default generation temperature |

---

## 5. Results

### 5.1 Generated Poetry Sample

Using the trained model with author conditioning, the following poem was generated with the seed "Свобода" (Freedom):

```
Свобода
По приказка на високия ден
само пред светлината си свети,
с полето става с твойто сводели
с пространството и моят мой път.
И в своя ден ще проповяда в тях
на полята си с очи на старият
скръб на странния полет пред мен
без мен в своя сън със своя свят син.
```

### 5.2 Observations

The generated text demonstrates:

1. **Syntactic Coherence:** The model produces grammatically plausible Bulgarian text
2. **Poetic Structure:** Maintains line breaks and poetic rhythm
3. **Thematic Consistency:** Related imagery (light, sky, flight, freedom)
4. **Style Conditioning:** The author embedding influences the generated style

---

## 6. Tests

The implementation was validated through a test script verifying all components:

| Test | Component | Result |
|------|-----------|--------|
| 1/4 | Dictionary creation | ✓ Passed |
| 2/4 | Model initialization | ✓ Passed |
| 3/4 | Forward pass (training mode) | ✓ Passed |
| 4/4 | Text generation | ✓ Passed |

---

## 7. Conclusions

### 7.1 Achieved Results

1. **Successful implementation** of a multi-layer LSTM language model with packed sequences
2. **Author conditioning** through learned embeddings for style-specific generation
3. **Temperature-controlled sampling** for varied text generation
4. **Training and evaluation** pipeline with perplexity metrics

### 7.2 Future Work

- Increase model capacity (more layers, larger hidden size)
- Experiment with attention mechanisms
- Fine-tune on specific authors
- Implement beam search for generation
- Add syllable-aware generation for better poetic meter

---

## 8. Technical Details

### 8.1 Requirements

```
Python >= 3.8
torch (PyTorch)
numpy
pickle
```

### 8.2 Project Structure

```
HW3/
├── README.md                 # This document
├── a3/
│   ├── model.py              # LSTM model implementation
│   ├── generator.py          # Text generation function
│   ├── train.py              # Training and perplexity functions
│   ├── run.py                # Main script
│   ├── parameters.py         # Hyperparameters
│   ├── utils.py              # Utility functions
│   ├── test_script.py        # Validation tests
│   ├── corpusPoems           # Training corpus
│   ├── trainData             # Preprocessed training data
│   ├── testData              # Preprocessed test data
│   ├── char2id               # Character vocabulary
│   ├── auth2id               # Author vocabulary
│   └── modelLSTM             # Trained model weights
└── solution/
    ├── model.py              # Final model implementation
    ├── generator.py          # Final generator implementation
    ├── parameters.py         # Final parameters
    ├── modelLSTM             # Trained model
    └── result-from-model-poem_freedom.txt  # Generated sample
```

### 8.3 Execution

```bash
# Create and activate virtual environment (first time only)
cd HW3
python3 -m venv venv
source venv/bin/activate
pip install torch numpy

# Navigate to working directory
cd a3

# Prepare data from corpus
python run.py prepare

# Train model
python run.py train

# Continue training from checkpoint
python run.py train modelLSTM

# Evaluate perplexity
python run.py perplexity

# Generate poem
python run.py generate "Никола Вапцаров" "{Свобода
"

# Generate with custom temperature
python run.py generate "Христо Ботев" "{" 0.6

# Run validation tests
python test_script.py
```

---

## References

1. Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation, 9(8), 1735-1780.
2. Graves, A. (2013). *Generating Sequences With Recurrent Neural Networks*. arXiv:1308.0850
3. Karpathy, A. (2015). *The Unreasonable Effectiveness of Recurrent Neural Networks*. Blog post.
4. Sutskever, I., Martens, J., & Hinton, G. E. (2011). *Generating Text with Recurrent Neural Networks*. ICML 2011.

---

*This document was created as part of Homework Assignment 3 for the course "Information Retrieval and Deep Learning", Faculty of Mathematics and Informatics, Sofia University "St. Kliment Ohridski", 2025/2026*
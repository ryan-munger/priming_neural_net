# Project setup
- setup local virtual environment for the project
    - `python -m venv priming-venv` 
    - `.\priming-venv\Scripts\activate` (windows) `source ./priming-venv/bin/activate` (Mac/ Linux) - activates venv
        - you should see a (venv) in your command line prompt
- install python dependencies from requirements.txt
    - `pip install -r requirements.txt`

# Perceptual Priming Neural Network

This project models **perceptual priming** using a dual-pathway neural network trained on common English words. It simulates how **prior exposure** to certain words (or visually similar words) can bias recognition in **word fragment completion tasks**. The model learns from both **word frequency** and **orthographic structure**, and supports **activation spreading** during priming.

---

## Overview

The model integrates two key cognitive components:

- **Word identity** (learned through frequency of exposure)
- **Orthographic similarity** (learned through character sequences)

It is implemented using **TensorFlow/Keras** and trained on a word list with frequency weighting. During inference, the model accepts priming inputs and boosts the activations of both the primed words and their **orthographic neighbors**.

---

## Architecture 

### Model Inputs

| Input Name | Description | Shape |
|------------|-------------|-------|
| `word_input` | Integer ID representing the word | `(batch_size, 1)` |
| `char_input` | Character-level encoding of the word | `(batch_size, max_word_length)` |

---

### Word Embedding Path

1. **Embedding Layer**
   - Learns an 8-dimensional vector for each word index.
   - Captures frequency and identity-based word representations.
   - _Shape_: `(batch_size, 1, 8)`

2. **Flatten Layer**
   - Converts `(1, 8)` to `(8,)` for merging with orthographic data.
   - _Shape_: `(batch_size, 8)`

---

### Character Embedding Path

1. **Character Embedding**
   - Learns a 4D vector for each character.
   - _Shape_: `(batch_size, max_word_length, 4)`

2. **LSTM Layer**
   - Processes character sequences to extract orthographic patterns.
   - Mimics visual form processing in human reading.
   - _Shape_: `(batch_size, 16)`

---

### Combination & Prediction

1. **Concatenation Layer**
   - Merges flattened word embedding and LSTM output.
   - _Shape_: `(batch_size, 24)`

2. **Dense + Softmax Output**
   - Outputs a probability distribution over the vocabulary.
   - Uses softmax activation.
   - _Shape_: `(batch_size, vocab_size + 1)`

---

## Cognitive Parallels

| Component | Simulated Cognitive Process |
|----------|------------------------------|
| Word embedding | Long-term word familiarity |
| Char-level LSTM | Visual word form encoding |
| Concatenation | Integration of identity + form |
| Logit boosting | Spreading activation from priming |
| Softmax | Lexical access & decision |

---

## Priming Simulation

During prediction:
- **Priming words** are given a logit boost to make them more competitive.
- **Orthographic neighbors** of primed words are also boosted (scaled by similarity).
- Only valid completions (can solve the word fragment) are considered.

---

## File Descriptions

- `train_model.py`: Trains the model.
- `model_predict.py`: Loads the model, accepts priming words, and completes word fragments.
- `training-word-set.txt`: The vocabulary list with frequency (via repetition) data.
- `word_tokenizer.pkl` / `char_mapping.pkl`: Saved tokenizers and mappings.
- `orthographic_word_model.keras`: The trained & saved model.

---

## Example Usage & Output

```bash
$ python train_model.py

$ python model_predict.py

Enter priming words separated by spaces: WENT, WAGE, TOOL

Orthographic neighbors of priming words:
  went,: went (0.80), west (0.60), rent (0.60), want (0.60), sent (0.60)
  wage,: wage (0.80), page (0.60), wake (0.60), wave (0.60), ages (0.60)
  tool: pool (0.75), cool (0.75), took (0.75), toll (0.75)


Enter 'exit' instead of a fragment to exit.

Enter Word Fragment (Ex. W_NT): W_NT
Possible solutions for fragment from word set: ['went', 'want']
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 120ms/step

Priming activation spread:
  wage,: +2.0 (direct prime)
  wage: +0.96 (orthographic neighbor)
  pool: +0.90 (orthographic neighbor)
  tool: +2.0 (direct prime)
  cool: +0.90 (orthographic neighbor)
  went: +0.96 (orthographic neighbor)
  went,: +2.0 (direct prime)
  toll: +0.90 (orthographic neighbor)
  took: +0.90 (orthographic neighbor)

Predicted word: WENT

Orthographic similarity of possible completions to priming words:
  went (PREDICTED): went,: 0.80, wage,: 0.20, tool: 0.00
  want: went,: 0.60, wage,: 0.40, tool: 0.00

Enter Word Fragment (Ex. W_NT): _OOL
Possible solutions for fragment from word set: ['tool', 'pool', 'cool']
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 100ms/step

Priming activation spread:
  wage,: +2.0 (direct prime)
  wage: +0.96 (orthographic neighbor)
  pool: +0.90 (orthographic neighbor)
  tool: +2.0 (direct prime)
  cool: +0.90 (orthographic neighbor)
  went: +0.96 (orthographic neighbor)
  went,: +2.0 (direct prime)
  toll: +0.90 (orthographic neighbor)
  took: +0.90 (orthographic neighbor)

Predicted word: TOOL

Orthographic similarity of possible completions to priming words:
  tool (PREDICTED): went,: 0.00, wage,: 0.00, tool: 1.00
  pool: went,: 0.00, wage,: 0.00, tool: 0.75
  cool: went,: 0.00, wage,: 0.00, tool: 0.75

Enter Word Fragment (Ex. W_NT): _AGE
Possible solutions for fragment from word set: ['page', 'wage']
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step

Priming activation spread:
  wage,: +2.0 (direct prime)
  wage: +0.96 (orthographic neighbor)
  pool: +0.90 (orthographic neighbor)
  tool: +2.0 (direct prime)
  cool: +0.90 (orthographic neighbor)
  went: +0.96 (orthographic neighbor)
  went,: +2.0 (direct prime)
  toll: +0.90 (orthographic neighbor)
  took: +0.90 (orthographic neighbor)

Predicted word: PAGE

Orthographic similarity of possible completions to priming words:
  page (PREDICTED): went,: 0.00, wage,: 0.60, tool: 0.00
  wage: went,: 0.20, wage,: 0.80, tool: 0.00

Enter Word Fragment (Ex. W_NT): exit
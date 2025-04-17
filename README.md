# Perceptual Priming Neural Network Model

## Overview

This repository contains a neural network model designed to simulate perceptual priming effects in word recognition. The model learns words based on their orthographic (visual) features and frequency, then demonstrates how prior exposure to certain words increases the activation of those words and orthographically similar words during subsequent word recognition tasks.

## Cognitive Science Background

Perceptual priming is a memory phenomenon where exposure to a stimulus influences response to a later stimulus, without conscious awareness. In word recognition, this means seeing a word makes it (and visually similar words) easier to identify later. This project implements several key cognitive science principles:

1. **Frequency Effects**: Words encountered more frequently are represented more strongly in memory and are recognized more easily (Broadbent, 1967; Murray & Forster, 2004)

2. **Orthographic Similarity**: Words sharing visual features with primed words also benefit from spreading activation (Coltheart et al., 2001)

3. **Priming as Residual Activation**: The model implements priming as residual activation that spreads through a neural network (McClelland & Rumelhart, 1981)

4. **Automatic vs. Conscious Processing**: The priming effect happens automatically, without explicit memory recall, matching how human implicit memory operates

## Model Architecture

The model consists of a neural network with the following structure:

### Input Layer
- **Word Input**: One-hot encoded representation of word identity
- **Character Input**: Character-level representation (capturing orthographic features)

This dual-input approach mirrors theories of parallel processing in word recognition (Seidenberg & McClelland, 1989), where both whole-word and sub-lexical features contribute to word identification.

### Embedding Layers
- **Word Embedding Layer**: 32-dimensional representation of word identity
- **Character Embedding Layer**: 16-dimensional representation of each character

These embedding layers create distributed representations that capture similarity relationships between words (Rogers & McClelland, 2004).

### Feature Processing
- **Flatten Layer**: For word embeddings
- **LSTM Layer (24 units)**: For character sequence processing

The LSTM layer processes character sequences and extracts orthographic patterns, similar to how the visual word form area in the brain processes letter patterns (Dehaene et al., 2005).

### Integration and Representation
- **Concatenation Layer**: Combines word identity and orthographic features
- **Dense Layer (64 units, ReLU)**: Initial integration of features
- **Dropout Layer (0.2)**: Improves generalization
- **Bottleneck Layer (24 units, ReLU)**: Creates a frequency-weighted representation

The bottleneck layer serves as the model's learned representation space, with activation patterns that reflect both word identity and frequency, similar to how words are represented in the brain's lexical processing networks (Pulvermüller, 1999).

### Output Layer
- **Dense Layer (vocabulary size, Softmax)**: Word identification output

## Training Process

The model is trained with sample weights proportional to word frequency, ensuring that high-frequency words develop stronger representations. This mirrors the frequency effects observed in human lexical processing, where common words are recognized faster and more accurately.

```python
# Apply log transformation to compress the range of frequencies
sample_weights = np.log1p(sample_weights)  # log(1+x) to handle counts of 1
sample_weights /= sample_weights.max()  # Normalize to [0,1]

# Scale weights more aggressively to emphasize frequency differences
sample_weights = sample_weights ** 2  # Square the weights to emphasize high frequency words
```

This implementation of frequency effects is consistent with power law relationships between word frequency and recognition speed in human subjects (Howes & Solomon, 1951).

## Priming Implementation

The priming mechanism works through several cognitive science-informed steps:

1. **Direct Activation Boost**: Previously seen words receive a direct activation boost, mimicking residual activation in neural pathways

2. **Spreading Activation**: Orthographically similar words receive proportional activation boosts based on:
   - Orthographic similarity (visual form)
   - Similarity in the learned representation space (embedding similarity)

3. **Combined Priming Effect**: The final priming effect combines direct and spreading activation, consistent with interactive activation models of word recognition (McClelland & Rumelhart, 1981)

```python
# Combined effect (weighting orthographic similarity higher for perceptual priming)
combined_effect = (0.4 * embedding_effect) + (0.6 * ortho_effect)

# Direct match gets maximum boost
if target_word in priming_words:
    priming_effects[target_word] = 3.0  # Direct prime
else:
    # Scale the effect for spreading activation
    priming_effects[target_word] = combined_effect * 1.5
```

## Usage

### Training the Model
```python
python train_model.py
```

This trains the model on your word set and saves:
- The main model (`priming_model.keras`)
- An embedding extraction model (`embedding_model.keras`)
- Tokenizers and mapping files

### Running Priming Simulations
```python
python priming_simulation.py
```

This allows you to:
1. Specify priming words
2. Enter word fragments (e.g., "W_RD")
3. See how priming affects possible completions
4. Visualize the embedding space and priming effects


## Cognitive Science Implications

This model demonstrates several important cognitive science principles:

1. **Distributed Representations**: Words are represented as patterns of activation across multiple units rather than localized representations

2. **Interactive Activation**: Activation flows bidirectionally between word and sub-word features

3. **Automaticity**: Priming occurs automatically without explicit recall

4. **Graded Effects**: Priming strength varies continuously based on similarity, rather than being all-or-nothing

5. **Frequency as Learning History**: The model's frequency effects emerge from its training history, similar to how humans learn language through repeated exposure

## References

- Broadbent, D. E. (1967). Word-frequency effect and response bias. *Psychological Review, 74(1)*, 1-15.
- Coltheart, M., Rastle, K., Perry, C., Langdon, R., & Ziegler, J. (2001). DRC: A dual route cascaded model of visual word recognition and reading aloud. *Psychological Review, 108(1)*, 204-256.
- Dehaene, S., Cohen, L., Sigman, M., & Vinckier, F. (2005). The neural code for written words: A proposal. *Trends in Cognitive Sciences, 9(7)*, 335-341.
- Howes, D. H., & Solomon, R. L. (1951). Visual duration threshold as a function of word-probability. *Journal of Experimental Psychology, 41(6)*, 401-410.
- McClelland, J. L., & Rumelhart, D. E. (1981). An interactive activation model of context effects in letter perception: Part I. An account of basic findings. *Psychological Review, 88(5)*, 375-407.
- Murray, W. S., & Forster, K. I. (2004). Serial mechanisms in lexical access: The rank hypothesis. *Psychological Review, 111(3)*, 721-756.
- Pulvermüller, F. (1999). Words in the brain's language. *Behavioral and Brain Sciences, 22(2)*, 253-279.
- Rogers, T. T., & McClelland, J. L. (2004). *Semantic cognition: A parallel distributed processing approach*. MIT Press.
- Seidenberg, M. S., & McClelland, J. L. (1989). A distributed, developmental model of word recognition and naming. *Psychological Review, 96(4)*, 523-568.

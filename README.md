
# Text Generation Using LSTM

This project implements a text generation model using Long Short-Term Memory (LSTM) networks in TensorFlow and Keras. The model is trained on a text corpus to predict the next word given a sequence of words. The training data is tokenized and sequences of words are fed to an LSTM-based neural network for prediction.

## Prerequisites

Before running the project, ensure you have the following installed:

- Python 3.x
- TensorFlow
- Numpy
- Pickle

You can install the required packages using the following command:

```bash
pip install tensorflow numpy pickle-mixin
```

## Files Included

- **main.txt**: The text file containing the data used to train the model.
- **token.pkl**: The tokenizer model saved as a pickle file after processing the text data.

## Steps to Run

### 1. Prepare Data

The text data from `main.txt` is read and processed. The text is tokenized and sequences of words are created, where each sequence contains 3 words and the next word to be predicted.

### 2. Train the Model

The model is trained using a Sequential model with the following layers:

- **Embedding Layer**: Converts input tokens into dense vectors of fixed size.
- **LSTM Layers**: Two LSTM layers with 512 and 1024 units respectively to capture dependencies in the sequence.
- **Dense Layers**: Fully connected layers to output predictions.

The model is compiled using `categorical_crossentropy` as the loss function and the Adam optimizer.

### 3. Predict the Next Word

Once trained, the model can predict the next word in a given sequence. You can enter a line of text, and the model will predict the next word based on the last 3 words in the input.

### 4. Example Input and Output

To run the prediction loop, simply execute the code and enter a sequence of words. The model will output the predicted next word.

Example:

```
Enter your line: I am going
['I', 'am', 'going']
to
```

If you want to exit the prediction loop, input `0`:

```
Enter your line: 0
Execution completed.....
```

## Usage

1. Clone or download this repository.
2. Place your text file (`main.txt`) in the root folder.
3. Run the following command to start training the model:

```bash
python your_script.py
```

4. After training is complete, you can enter text lines to predict the next word.

## Model Architecture

- **Embedding Layer**: Embeds the input word sequences into a continuous vector space.
- **LSTM Layers**: Two layers of LSTMs to capture temporal patterns in the data.
- **Dense Layers**: Fully connected layers for outputting predictions.

The LSTM model's final layer uses the `softmax` activation function to predict the most probable next word.

## Tokenizer

The tokenizer is saved as `token.pkl` after being trained on the text corpus. It is used for converting the input text into sequences that the model can understand and process.

## Future Improvements

- **Model Tuning**: More epochs or fine-tuning of LSTM layers can improve accuracy.
- **Data Augmentation**: Larger text datasets can help the model generalize better.
- **Word Embeddings**: Pretrained embeddings such as GloVe or Word2Vec could be incorporated.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

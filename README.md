### What is Natural Language Processing (NLP) 

> Natural language processing, which evolved from computational linguistics, uses methods from various disciplines, such as computer science, artificial intelligence, linguistics, and data science, to enable computers to understand human language in both written and verbal forms. 
[IBM](https://www.ibm.com/blogs/watson/2020/11/nlp-vs-nlu-vs-nlg-the-differences-between-three-natural-language-processing-concepts/)

The usage of computers to have them the ability to understand both written and verbal forms in this case, text.
Types of NLP:
* Speech Recognition
* Machine Translation
* Sentiment Analysis
* Semantic Search

### What we're going to do?

Going to try and predict whether a tweet resembles a disaster tweet or not using NLP with various models and experimentations which would help in our classification.

Dataset Taken from [Kaggle's Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started) [PubMed 200k RCT dataset](https://github.com/Franck-Dernoncourt/pubmed-rct)
---
**Tokenization** is straight up mapping the words without any weight/values, just regular numerical encoding  
  
* **Word-Level Tokenization** - maps the whole text and maps each word; thus it(the word) is considered a token. (e.q. one-hot encoding)
* **Character-Level Tokenization** - maps the whole text but focuses on each letter from 1 - 26; thus it(each letter) is considered a token.  
* **Sub-word Tokenization** - takes the syllables of a word and tokenizes it 

**Embedding** uses vector weights that can be learned as out network trains. i.e. individual letters in a word on how each letter can be of importance to create that word.

---

🔑 **Takeaways and Conclusions**  

> When stacking use **`return_sequences`**, it basically returns its **timesteps/feature** from the embedding layer. Based from the documentation, the LSTM model takes in **3 inputs `[batch, timesteps, feature]`** not 2, if we don't retain the layer we'll be having an error with its shape when passing it to the next LSTM layer.

> when building your own model from scratch, again never forget to convert them into of numerical value, and in NLP's case embed them to be more of a vector as a vector -it can be of weighted value, or has patterns to be learned by our model

`LSTM` - Long Short Term Memory, its like logic gates where it stores 1s and 0s.
> see it this way, when you feed it an training set, it learns its weights and uses those patterns to predict the next word, it can just store depending on how the value of the weight performs based on the tanh activation function.

[Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM)](https://www.youtube.com/watch?v=WCUNPb-5EYI&t=297s&ab_channel=BrandonRohrer)

* From Model 1-5 our models seem to be overfitting the train data,  

> we can try using or feeding them more data, or use these models on a dataset that's more complicated
 since our dataset is relatively small, a simpler model would be a better fit for this or a transfer learning model


> Transfer Learning also works with less data, even with less data it can find patterns.
feeding it to a large model, if it tends to overfit give it more data, or simplify the model
  
   
  
**Process**  

* **Build a Text Vectorizer**  
    * **`max_vocab_length`** - 1000  
    * **`max_length`** - taken from the length of each word in a sentence, summed together  
and divided by the total amount of train_sentences  `sum([len(i.split()) for i in train_]) / len(train_)`
 

* **Build a Embedding Layer**
    * **`input_dim`** - same as `max_vocab_length`  
    * **`output_dim`** - any number divisible by `8`  
    * **`input_length`** - same as `max_length`  
    
     
> You must vectorize your text before feeding it to the embedding layer

### Go to   
[Experimentation Notebook](https://github.com/jjaimwork/RNN-Natural-Language-Processing/blob/master/Milestone%20Project%20Experimentation.ipynb)    for more of WorkFlow

[Milestone Notebook](https://github.com/jjaimwork/RNN-Natural-Language-Processing/blob/master/Milestone%20Project%20%5BExplained%5D.ipynb)   for the Conclusion and also a step by step overview of what we've done

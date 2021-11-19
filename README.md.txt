### What is Natural Language Processing (NLP) 

> Natural language processing, which evolved from computational linguistics, uses methods from various disciplines, such as computer science, artificial intelligence, linguistics, and data science, to enable computers to understand human language in both written and verbal forms. 
[IBM](https://www.ibm.com/blogs/watson/2020/11/nlp-vs-nlu-vs-nlg-the-differences-between-three-natural-language-processing-concepts/)

The usage of computers to have them the ability to understand both written and verbal forms in this case, text.
Types of NLP:
* Speech Recognition
* Machine Translation
* Sentiment Analysis
* Semantic Search


---
**Tokenization** is straight up mapping the words without any weight/values, just regular numerical encoding  
  
* **Word-Level Tokenization** - maps the whole text and maps each word; thus it(the word) is considered a token. (e.q. one-hot encoding)
* **Character-Level Tokenization** - maps the whole text but focuses on each letter from 1 - 26; thus it(each letter) is considered a token.  
* **Sub-word Tokenization** - takes the syllables of a word and tokenizes it 

**Embedding** uses vector weights that can be learned as out network trains. i.e. individual letters in a word on how each letter can be of importance to create that word.

---

🔑 **Takeaways**  
  
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
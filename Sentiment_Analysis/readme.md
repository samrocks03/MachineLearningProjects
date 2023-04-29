<h1>Sentiment Analysis of Amazon Fine Food Reviews</h1>
  In this project, we will perform sentiment analysis on the Amazon Fine Food Reviews dataset using two different methods:

<ol>
<li>Using VADER (Valence Aware Dictionary and sEntiment Reasoner), a rule-based sentiment analysis tool.
<li>Using a pre-trained transformer model called twitter-roberta-base-sentiment, which is a fine-tuned version of the RoBERTa model.
Dataset
  </ol>
  
The [Amazon Fine Food Reviews dataset](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) contains over 500,000 reviews of food products sold on Amazon from 1999 to 2012. The reviews include information about the product, the reviewer, and the text of the review, as well as a score ranging from 1 to 5.

<h2> Code </h2>
The code for this project can be found in the [Jupyter notebook available] on GitHub.

The notebook uses Python, and the following packages:
* numpy
* pandas
* matplotlib
* seaborn
* nltk
* transformers
* tqdm
* scipy


### Methodology

####  VADER
  The first method of sentiment analysis used in this project is VADER (Valence Aware Dictionary and sEntiment Reasoner), a rule-based sentiment analysis tool. VADER uses a lexicon of words and phrases that have been rated on a scale from -4 (most negative) to +4 (most positive), and combines these ratings with rules to determine the overall sentiment of a piece of text.

The notebook first uses VADER to calculate the sentiment scores of each review in the dataset, and then plots the distribution of compound sentiment scores for each score rating (1 to 5).

####  RoBERTa

<p>The second method used in this project is a pre-trained transformer model called twitter-roberta-base-sentiment, which is a fine-tuned version of the RoBERTa model. This model was trained on tweets and has three possible outputs: negative, neutral, and positive.</p>

<p>The notebook first uses this model to calculate the sentiment scores of a single example review, and then applies it to the entire dataset. The notebook then plots the distribution of sentiment scores for each score rating (1 to 5) separately for each sentiment label (negative, neutral, and positive).</p>


<h2> Conclusion </h2>
Overall, this project provides an interesting analysis of the sentiment of Amazon Fine Food Reviews using two different methods: a rule-based approach using VADER and a machine learning-based approach using a pre-trained transformer model. The results suggest that both methods are capable of producing useful insights into the sentiment of text data.

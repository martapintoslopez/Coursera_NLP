import re
import string
import numpy as np
# Natural Language Toolkit https://www.nltk.org/
import nltk                                
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize    # module for tokenizing strings

#We are going to use a tokenizer that requires the Punkt sentence tokenization models to be installed.
nltk.download('punkt')
# download the stopwords from NLTK
nltk.download('stopwords')

def preprocessing(row):
    """Process row function.
    Input:
        row: a string containing a row
    Output:
        rows_clean: a list of words containing the processed row
    """
    # Convert to a string
    row = str(row)

    # Import stopwords de NLTK
    stopwords_row = stopwords.words()
    
    stemmer = PorterStemmer()
    
    # remove hyperlinks
    row2 = re.sub(r'https?://[^\s\n\r]+', '', row)

    # tokenize row
    clean_row = word_tokenize(row2)

    rows_clean = []
    for word in clean_row:
        if (word not in stopwords_row and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            stem_word = stemmer.stem(word)  # stemming word
            rows_clean.append(stem_word)  # add word to rows_clean

    return rows_clean

def build_freqs(rows, ys):
    """Build frequencies.
    Input:
        rows: a list of rows
        ys: an m x 1 array with the sentiment label of each tweet
            (either 0 or 1)
    Output:
        freqs: a dictionary mapping each (word, sentiment) pair to its
        frequency
    """
    # Convert np array to list since zip needs an iterable.
    # The squeeze is necessary or the list ends up with one element.
    # Also note that this is just a NOP if ys is already a list.
    yslist = np.squeeze(ys).tolist()

    # Start with an empty dictionary and populate it by looping over all rows
    # and over all processed words in each tweet.
    freqs = {}
    for y, row in zip(yslist, rows):
        for word in preprocessing(row):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1

    return freqs

def sigmoid(z): 
    '''
    Input:
        z: is the input (can be a scalar or an array)
    Output:
        h: the sigmoid of z
    '''
    # calculate the sigmoid of z
    h = 1 / (1 + np.exp(-z))
    
    return h

def gradientDescent(x, y, theta, alpha, num_iters):
    '''
    Input:
        x: matrix of features which is (m,n+1)
        y: corresponding labels of the input matrix x, dimensions (m,1)
        theta: weight vector of dimension (n+1,1)
        alpha: learning rate
        num_iters: number of iterations you want to train your model for
    Output:
        J: the final cost
        theta: your final weight vector
    '''
    # the number of rows in matrix x
    m = x.shape[0]
    
    for i in range(0, num_iters):
        z = np.dot(x, theta)
        h = sigmoid(z)
        
        # the cost function
        J = -1./m * (np.dot(y.T, np.log(h)) + np.dot((1-y).T, np.log(1-h)))
        # update the weights theta
        theta = theta - ((alpha/m) * np.dot(x.T, (h-y)))
        
    J = float(J)
    return J, theta

def extract_features(row, freqs):
    '''
    Input: 
        row: a string containing ONE title
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
    Output: 
        x: a feature vector of dimension (1,3)
    '''
    # preprocessing tokenizes, stems, and removes stopwords
    word_list = preprocessing(row)
    
    # 3 elements for [bias, positive, negative] counts
    x = np.zeros(3) 
    # bias term is set to 1
    x[0] = 1 
      
    for word in word_list:
        
        # increment the word count for the positive label 1
        x[1] += freqs.get((word, 1.0),0)
        # increment the word count for the negative label 0
        x[2] += freqs.get((word, 0.0),0)
    
    x = x[None, :]  # adding batch dimension for further processing
    assert(x.shape == (1, 3)) # If the shape of x is not (1, 3), the assertion raises an AssertionError
    return x

def predict_title(title, freqs, theta):
    '''
    Input: 
        tweet: a string
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
        theta: (3,1) vector of weights
    Output: 
        y_pred: the probability of a tweet being positive or negative
    '''
    t = extract_features(title, freqs)
    # Probability of a course being science or humanities
    y_pred = sigmoid(np.dot(t,theta))

    return y_pred

def test_logistic_regression(test_x, test_y, freqs, theta):
    """
    Input: 
        test_x: a list of rows
        test_y: (m, 1) vector with the corresponding labels for the list of rows
        freqs: a dictionary with the frequency of each pair (or tuple)
        theta: weight vector of dimension (3, 1)
    Output: 
        accuracy: (# of rows classified correctly) / (total # of rows)
    """
    # the list for storing predictions
    y_hat = []
    
    for title in test_x:
        # get the label prediction for the tweet
        y_pred = predict_title(title, freqs, theta)
        
        if y_pred > 0.5:
            # append 1.0 to the list
            y_hat.append(1.0)
        else:
            # append 0 to the list
            y_hat.append(0.0)

    # y_hat is a list, but test_y is (m,1) array so we convert both to one-dimensional array in order to compare them
    accuracy = (y_hat == np.squeeze(test_y)).sum()/len(test_x)
    
    return accuracy

def lookup(freqs, word, label):
    '''
    Input:
        freqs: a dictionary with the frequency of each pair (or tuple)
        word: the word to look up
        label: the label corresponding to the word
    Output:
        n: the number of times the word with its corresponding label appears.
    '''
    n = 0  # freqs.get((word, label), 0)
    pair = (word, label)
    if (pair in freqs):
        n = freqs[pair]

    return n

def train_naive_bayes(freqs, train_x, train_y):
    '''
    Input:
        freqs: dictionary from (word, label) to how often the word appears
        train_x: a list of tweets
        train_y: a list of labels corresponding to the tweets (0,1)
    Output:
        logprior: the log prior. (equation 3 above)
        loglikelihood: the log likelihood of you Naive bayes equation. (equation 6 above)
    '''
    loglikelihood = {}
    logprior = 0

    # V: the number of unique words in the vocabulary
    vocabulary = set(pair[0] for pair in freqs.keys())
    V = len(vocabulary)
    print("La cantidad de palabras únicas es:", V)

    # N_sci, N_neg, V_sci, V_neg
    N_sci = V_sci = N_hum = V_hum =0
    for pair in freqs.keys():
        if pair[1] > 0:
            # Increment the number of science words by the count for this (word, label) pair
            N_sci += freqs[pair]
        else:
            # Increment the number of humanities words by the count for this (word, label) pair
            N_hum += freqs[pair]

    # D: the number of courses
    D = len(train_x)
    print("El número de cursos es: ", D)

    # D_sci: the number of science courses (train_y ==1.0)
    D_sci = (len(list(filter(lambda x: x > 0, train_y))))
    # D_hum: the number of humanities courses (train_y ==1.0)
    D_hum = (len(list(filter(lambda x: x <= 0, train_y))))

    # logprior
    logprior = np.log(D_sci) - np.log(D_hum)

    # log likelihood of any word
    for word in vocabulary:
        #frequency of a word in science or humanities
        freq_sci = lookup(freqs, word, 1)
        freq_hum = lookup(freqs, word, 0)

        #Probability of each word to be in science or humanities context
        prob_sci = (freq_sci + 1) / (N_sci + V)
        prob_hum = (freq_hum + 1) / (N_hum + V)

        # Calculate log likelihood of each word
        loglikelihood[word] = np.log(prob_sci / prob_hum)
    
    return logprior, loglikelihood
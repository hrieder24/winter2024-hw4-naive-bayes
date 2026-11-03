import math
import re

class Bayes_Classifier:

    def __init__(self):
        #postive (5-star) or negative (1-star)
        #dictionary w/ two keys mapping to sep dicts for one star and another for five star
        #records how often each word appears in positive and negative reviews
        self.word_freqs = {'1': {}, '5': {}}
        #dictionary w/ two keys
        #total number of positive and negative reviews
        self.rev_count = {'1': 0, '5': 0}
        self.total_revs = 0

    def process(self, text):
        #make all text lowercase, replace punctuation/symbol w/ a space, 
        #and then split texts into indiv.
        text = text.lower()
        text = re.sub(r'\W', ' ', text)
        tokens = text.split()
        return tokens
    
    def train(self, lines):
        for line in lines:
            #split each line into stars rating, ID (ignored), and the review text
            stars, _, text = line.partition('|')
            #process the reviews text using helper function
            tokens = self.process(text)
            #add one to the count of reviews 
            self.rev_count[stars] += 1
            #for each word/"token"
            for token in tokens:
                #if its not already in dict for pos or neg class, then add it and incr count by one
                if token not in self.word_freqs[stars]:
                    self.word_freqs[stars][token] = 1
                #otherwise its there so just add to its count
                else:
                    self.word_freqs[stars][token] += 1
        #add all the reviews that were processed to total count            
        self.total_revs = sum(self.rev_count.values())

    def calculate_prob(self, tokens, class_): #use smoothing**
        #prior probability for class
        log_prob = math.log(self.rev_count[class_] / self.total_revs)
        total_words = sum(self.word_freqs[class_].values())
        #set w/ all unique words in pos/neg dictionaries
        vocab = set(word for words in self.word_freqs.values() for word in words)
        vocab_size = len(vocab)
        for token in tokens:
            #get freq of token, if not present then freq is 0
            word_freq = self.word_freqs[class_].get(token, 0)
            #using add-one smoothing
            word_prob = (word_freq + 1) / (total_words + vocab_size)
            #add its log to the total log probability
            log_prob += math.log(word_prob)
            #####GET OFFICE HOURS HELP######
        return log_prob
    
    def classify(self, lines):
        predictions = []
        for line in lines:
            tokens = self.process(line)
            #calculate the logprob for each class based on the tokens
            probs = {class_: self.calculate_prob(tokens, class_) for class_ in self.rev_count}
            #find class w/ max prob & add this prediction to list
            predictions.append(max(probs, key=probs.get))
        return predictions

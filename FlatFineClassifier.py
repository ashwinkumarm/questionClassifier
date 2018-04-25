import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tag.stanford import StanfordNERTagger
from scipy.sparse import hstack
from sklearn.svm import LinearSVC
from practnlptools.tools import Annotator
from readproperties import read_property
import naiveBayes

# #removing special characters from sentence##


def preprocess(raw_sentence):
    sentence = re.sub(r'[$|.|!|"|(|)|,|;|`|\']', r'', raw_sentence)
    return sentence

# #making the file format ready to use##


def file_preprocess(filename):
    corpus = []
    classes = []
    f = open(filename, 'r')
    lines = f.readlines()
    for line in lines:
        line = line.rstrip('\n')
        if not (line == "\n"):
            classes.append((line.split()[0]).split(":")[1])
    for line in lines:
        line = line.rstrip('\n')
        line = preprocess(line)
        sentence = ""
        words = line.split()
        for i in range(0, len(words)):
            if not(i == 0):
                sentence = sentence + (words[i]) + " "
        corpus.append(sentence)
    f.close()
    return corpus, classes

# #Compute POS##


def compute_POS_Tags(corpus):
    POS = []
    f = open(read_property('POS_features_test_coarse_path'), "r")
    corpus = []
    for lines in f:
        l = lines.split()
        words = ""
        for w in l:
            words = words + w + " "
        POS.append(words)   
    return POS         
      
# #Compute NER## 

    
def compute_NER(corpus):
    NER = []
    f = open(read_property('NER_features_test_coarse_path'), "r")
    for lines in f:
        l = lines.split()
        words = ""
        for w in l:
            words = words + w + " "
        NER.append(words)        
    return NER

# #Compute Chunks##  

   
def compute_Chunks(corpus):
    f = open(read_property('Chunk_features_test_path'), "r")
    chunks = []
    for lines in f:
        l = lines.split()
        words = ""
        for w in l:
            words = words + w + " "
        chunks.append(words)  
    return chunks      


def compute_REL_Tags(corpus):
    POS = []
    f = open(read_property('REL_features_test_coarse_path'), "r")
    corpus = []
    for lines in f:
        l = lines.split()
        words = ""
        for w in l:
            words = words + w + " "
        POS.append(words)   
    return POS      

######################################TRAINING############################################

#######Train class labels#####


train_class = []
f = open(read_property('trainingfilepath'), 'r')
lines = f.readlines()
for line in lines:
    line = line.rstrip('\n')
    if not (line == "\n"):
        train_class.append((line.split()[0]).split(":")[1])

###words in question###

print "Training"
f = open(read_property('word_features_train_coarse_path'), "r")
corpus = []
for lines in f:
    l = lines.split()
    words = ""
    for w in l:
        words = words + w + " "
    corpus.append(words)        
vectorizer_words = CountVectorizer(min_df=1)
X_words = vectorizer_words.fit_transform(corpus)
f.close()
print "word feature extraction done"

###POS tags in question###

f = open(read_property('POS_features_train_coarse_path'), "r")
corpus = []
for lines in f:
    l = lines.split()
    words = ""
    for w in l:
        words = words + w + " "
    corpus.append(words)        
vectorizer_POS = CountVectorizer(min_df=1)
X_POS = vectorizer_POS.fit_transform((corpus))  
f.close()
print "POS feature extraction done"

###NER tags in question###

f = open(read_property('NER_features_train_coarse_path'), "r")
corpus = []
for lines in f:
    l = lines.split()
    words = ""
    for w in l:
        words = words + w + " "
    corpus.append(words)        
vectorizer_NER = CountVectorizer(min_df=1)
X_NER = vectorizer_NER.fit_transform((corpus))  
print "Vectorize"
f.close()
print "NER feature extraction done"

###Chunk tags in question###

f = open(read_property('Chunk_features_train_path'), "r")
corpus = []
for lines in f:
    l = lines.split()
    words = ""
    for w in l:
        words = words + w + " "
    corpus.append(words)        
vectorizer_Chunk = CountVectorizer(min_df=1)
X_Chunk = vectorizer_Chunk.fit_transform((corpus))  
f.close()
print "Chunk feature extraction done"

f = open(read_property('REL_features_train_coarse_path'), "r")
corpus = []
for lines in f:
    l = lines.split()
    words = ""
    for w in l:
        words = words + w + " "
    corpus.append(words)        
vectorizer_REL = CountVectorizer(min_df=1)
X_REL = vectorizer_REL.fit_transform((corpus))  
f.close()
print "REL feature extraction done"

X = hstack((X_words, X_POS))
X_intermediate = hstack((X, X_REL))
X_train = hstack((X_intermediate, X_NER))
X_train = hstack((X_train, X_Chunk))

######################################TESTING############################################

print "In Testing"
filename_test = read_property('testfilepath')
corpus_test, test_class_gold = file_preprocess(filename_test)

###words in question###

X_words = vectorizer_words.transform(corpus_test)
print "Word features extracted"

###POS tags in question###

X_POS = vectorizer_POS.transform(compute_POS_Tags(corpus_test))     
print "POS features extracted"

###NER tags in question###

X_NER = vectorizer_NER.transform(compute_NER(corpus_test))  
print "NER features extracted" 

###Chunk tags in question###

X_Chunk = vectorizer_Chunk.transform(compute_Chunks(corpus_test))  
print "Chunk features extracted"

X_REL = vectorizer_REL.transform(compute_REL_Tags(corpus_test))
print "REL features extracted"

X = hstack((X_words, X_POS))
X_intermediate = hstack((X, X_REL))
X_test = hstack((X_intermediate, X_NER))
X_test = hstack((X_test, X_Chunk))

###################Applying the LinearSVC Classifier#########################

naiveBayes.naiveBayes(X_train, train_class, X_test, test_class_gold)

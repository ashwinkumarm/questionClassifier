'''
Created on 23-Apr-2018

@author: Ashwin
'''

from nltk.corpus import wordnet as wn
from readproperties import read_property
import re
import nltk


##removing special characters from sentence##
def preprocess(raw_sentence):
    sentence= re.sub(r'[$|.|!|"|(|)|,|;|`|\']',r'',raw_sentence)
    return sentence

##making the file format ready to use##
def file_preprocess(filename):
    corpus=[]
    classes=[]
    f=open(filename,'r')
    fi=open(read_property('word_features_train_coarse_path'),"w")
    lines=f.readlines()
    for line in lines:
        line=line.rstrip('\n')
        line=preprocess(line)
        #print "The line is  ",line
        sentence=""
        words=line.split()
        for i in range(0,len(words)):
            if not(i==0):
                sentence=sentence+(words[i])+" "
        fi.write(sentence+"\n")
        corpus.append(sentence)
    f.close()
    fi.close()
    return corpus,classes


def compute_RelWord(corpus):
    fi=open(read_property('REL_features_train_coarse_path'),"w")
    i = 0;
    for sentence in corpus:
        i = i+1
        if i == 1000:
            print("1000 completed")
            i = 0
        text = nltk.word_tokenize(sentence)
        rel_tags=""
        for word in text:
            if len(wn.synsets(word)) > 0:
                name = str(wn.synsets(word)[0].lemmas()[0].name())
            else:
                name = word 
            rel_tags=rel_tags+name +" "
        fi.write(rel_tags+"\n")
    fi.close()


filename_train=read_property('trainingfilepath')
print('read filename')
corpus,train_class=file_preprocess(filename_train)
print('file preprocess completed')
compute_RelWord(corpus)
print("similar word for each word is computed")
# QUestion Classifier

### The program finds the accuracy for coarse and fine classifier for TREC data set. We find accuracy
### for single prediction and multiple classes where classes are less than 5. 

Each program takes several hours to run. So we saved everything in files. 

extractclasstrainingtext - extracts coarse classifier and save in different files
naiveBayes - our model to predict top 5 classes
RelWord - TO extract the Relational words
Training_coarse - it computes and stores NER,POS,CHUNKS,WORDS and saves for training data set.
Training_coarse_classificatoin - Reads the training classified and predicts for test and stores and prints the accuracy for coarse classification
Training_fine - Extracts data for fine
Training_fine_Classification - Reads the training and predicts fine labels and stores the output and prints accuracy 
FlatFineClassifier - For classifying fine, implemented flat architecture instead of hierarchial

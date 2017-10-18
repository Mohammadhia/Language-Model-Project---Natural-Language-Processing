import nltk
from nltk.corpus import udhr
import re

#Cleaning up each language corpus by removing punctuation, making all lower case, etc.
english = udhr.raw('English-Latin1')
english = re.sub(r'[();,.:]', '', english)
english = english.replace("\n", " ").lower()

french = udhr.raw('French_Francais-Latin1')
french = re.sub(r'[();,.:]', '', french)
french = french.replace("\n", " ").lower()

italian = udhr.raw('Italian_Italiano-Latin1')
italian = re.sub(r'[();,.:]', '', italian)
italian = italian.replace("\n", " ").lower()

spanish = udhr.raw('Spanish_Espanol-Latin1')
spanish = re.sub(r'[();,.:]', '', spanish)
spanish = spanish.replace("\n", " ").lower()

#Training and development samples
english_train, english_dev = english[0:1000], english[1000:1100]
english_train = english_train.split() #Tokenizing training data
english_dev = english_dev.split() #Tokenizing dev set

french_train, french_dev = french[0:1000], french[1000:1100]
french_train = french_train.split() #Tokenizing training data
french_dev = french_dev.split() #Tokenizing dev set

italian_train, italian_dev = italian[0:1000], italian[1000:1100]
italian_train = italian_train.split() #Tokenizing training data
italian_dev = italian_dev.split() #Tokenizing dev set

spanish_train, spanish_dev = spanish[0:1000], spanish[1000:1100]
spanish_train = spanish_train.split() #Tokenizing training data
spanish_dev = spanish_dev.split() #Tokenizing dev set

#Test sets (first 1000 words in each language set)
english_test = english.split()
english_test = english_test[0:1000]

french_test = french.split()
french_test = french_test[0:1000]

italian_test = italian.split()
italian_test = italian_test[0:1000]

spanish_test = spanish.split()
spanish_test = spanish_test[0:1000]

#The following function converts a given UNIGRAM from counts to conditional probabilities
def unigramCountsToProbabilities(inputUnigram):
    total = 0 #Total number of character occurrences
    tempUnigram = inputUnigram #Holds copy of unigram that will be modified
    for key, value in inputUnigram.items():
        total += value  #Counts up total number of characters in unigram
    for key, value in tempUnigram.items():
        tempUnigram[key] = value / float(total) #Modifies each unigram from count to probability

    return tempUnigram  #Returns updated unigram

#The following function converts a given BIGRAM OR TRIGRAM (doesn't work with unigrams) from counts to conditional probabilities
def countsToProbabilities(ngram):
    conditionalSumDictionary = {} #Holds the total number of times each conditional appears (Ex: '<s>a': 11 means that 11 conditionals involve '<s>a' in a trigram
    tempNGram = ngram #Holds copy of ngram that will be modified
    for key, value in ngram.items():
        if(key[0] not in conditionalSumDictionary): #If the first element of the tuple (the conditional) is not in the dictionary
            conditionalSumDictionary[key[0]] = value
        else:
            conditionalSumDictionary[key[0]] += value
    for key, value in tempNGram.items():
        tempNGram[key] = value / float(conditionalSumDictionary.get(key[0])) #Gets conditional probability by dividing count by total number of times condition appears

    return tempNGram #Returns updated bigram or trigram

#The following function makes (trains) a unigram, bigram, and trigram for a given language's training data
def train(language_train):
    unigram, bigram, trigram = {}, {}, {} #Will hold the trained n-gram models

    for i in range(0, len(language_train)): #Loop through each word
        for j in range(0, len(language_train[i])):   #Loop through each character in word
            #Builds unigram model for english below (COUNTS ONLY)
            if(language_train[i][j] not in unigram):
                unigram[language_train[i][j]] = 1
            else:
                unigram[language_train[i][j]] += 1

            #Builds bigram model for english below (COUNTS ONLY)
            #NOTE: THE TUPLES ARE STRUCTURED SUCH THAT THE PREVIOUS CHARACTER IS LISTED FIRST, THEN THE SUBSEQUENT CHARACTER
            if(j != 0): #General rules; If not at the first character of the word (This also works for words that are longer than 1-letter)
                if ((language_train[i][j - 1], language_train[i][j]) not in bigram):
                    bigram[(language_train[i][j - 1], language_train[i][j])] = 1
                else:
                    bigram[(language_train[i][j - 1], language_train[i][j])] += 1
            elif(j == 0): #At the first character of the word
                if(('<s>', language_train[i][j]) not in bigram):
                    bigram[('<s>', language_train[i][j])] = 1
                else:
                    bigram[('<s>', language_train[i][j])] += 1
            #A CHARACTER MAY SATISFY BOTH THE FIRST BIGRAM CONDITIONAL AND THE ONE BELOW
            if(j == len(language_train[i]) - 1): #At the last character of the word
                if((language_train[i][j], '</s>') not in bigram):
                    bigram[(language_train[i][j], '</s>')] = 1
                else:
                    bigram[(language_train[i][j], '</s>')] += 1

        #Builds trigram model for english below (ONLY COUNTS)
        for j in range(0, len(language_train[i])):  # Loop through each character in word
            if(len(language_train[i]) == 1): #If a 1-letter word
                if(('<s>'+language_train[i][0], '</s>') not in trigram):
                    trigram[('<s>'+language_train[i][0], '</s>')] = 1
                else:
                    trigram[('<s>'+language_train[i][0], '</s>')] += 1
            if(len(language_train[i]) != 1 and j > 1):  #General rules; If not at the second character of the word AND the word is longer than 1 character
                if((language_train[i][j-2]+language_train[i][j-1], language_train[i][j]) not in trigram):
                    trigram[(language_train[i][j-2]+language_train[i][j-1], language_train[i][j])] = 1
                else:
                    trigram[(language_train[i][j - 2] + language_train[i][j - 1], language_train[i][j])] += 1
            elif(j == 1):   #At the second character of the word
                if(('<s>'+language_train[i][0], language_train[i][j]) not in trigram):
                    trigram[('<s>' + language_train[i][0], language_train[i][j])] = 1
                else:
                    trigram[('<s>' + language_train[i][0], language_train[i][j])] += 1
            if(j == len(language_train[i]) - 1 and j != 0): #At the last character of the word (AND WORD IS LONGER THAN 1 LETTER)
                if((language_train[i][j-1]+language_train[i][j], '</s>') not in trigram):
                    trigram[(language_train[i][j-1]+language_train[i][j], '</s>')] = 1
                else:
                    trigram[(language_train[i][j - 1] + language_train[i][j], '</s>')] += 1

    return unigramCountsToProbabilities(unigram), countsToProbabilities(bigram), countsToProbabilities(trigram) #Returns the n-gram models after converting from counts to probabilities

#The following function takes in 2 unigrams and a test set, and returns an array with the most probable language for each word in the test set
def compareUnigramAccuracy(language1, language2, test_set):
    resultArray = [] #Will hold the predicted language for each word (based on whichever unigram has highest probability)
    for i in range(0, len(test_set)): #for each word in the array
        language1Prob = 1 #Will hold probability of word being from language 1 (which should be the same language as the test set)
        language2Prob = 1 #Will hold probability of word being from language 2
        for j in range(0, len(test_set[i])): #for each character in the word
            if(test_set[i][j] in language1):
                language1Prob *= language1.get(test_set[i][j])
            else:
                language1Prob *= 0.0001 #Default small probability if character is not in the unigram
            if(test_set[i][j] in language2):
                language2Prob *= language2.get(test_set[i][j])
            else:
                language2Prob *= 0.0001 #Default small probability if character is not in the unigram
        if(language1Prob > language2Prob): #If the first language has the HIGHER probability of producing the word
            resultArray.append("language 1")
        elif(language1Prob == language2Prob):
            resultArray.append("language 1&2")  #If BOTH languages have EQUALLY likely chance of producing the word
        else:
            resultArray.append("language 2")    #If the first language has the LOWER probability of producing the word
    return resultArray #resultArray will be used to calculate accuracy

#The following function takes in 2 bigrams and a test set, and returns an array with the most probable language for each word in the test set
def compareBigramAccuracy(language1, language2, test_set):
    resultArray = []  # Will hold the predicted language for each word (based on whichever bigram has highest probability)
    for i in range(0, len(test_set)): #for each word in the array
        language1Prob = 1  # Will hold probability of word being from language 1 (which should be the same language as the test set)
        language2Prob = 1  # Will hold probability of word being from language 2
        for j in range(0, len(test_set[i])): #for each character in the word
            if(j != 0): #General rules; If not at the first character of the word (This also works for words that are longer than 1-letter)
                if((test_set[i][j - 1], test_set[i][j]) in language1):
                    language1Prob *= language1.get((test_set[i][j - 1], test_set[i][j]))
                else:
                    language1Prob *= 0.007 #Default small probability if the conditional is not in the bigram
                if ((test_set[i][j - 1], test_set[i][j]) in language2):
                    language2Prob *= language2.get((test_set[i][j - 1], test_set[i][j]))
                else:
                    language2Prob *= 0.007 #Default small probability if the conditional is not in the bigram
            elif(j == 0): #At the first character of the word
                if(('<s>', test_set[i][j]) in language1):
                    language1Prob *= language1.get(('<s>', test_set[i][j]))
                else:
                    language1Prob *= 0.007 #Default small probability if the conditional is not in the bigram
                if (('<s>', test_set[i][j]) in language2):
                    language2Prob *= language2.get(('<s>', test_set[i][j]))
                else:
                    language2Prob *= 0.007 #Default small probability if the conditional is not in the bigram
            if(j == len(test_set[i]) - 1): #At the last character of the word
                if((test_set[i][j], '</s>') in language1):
                    language1Prob *= language1.get((test_set[i][j], '</s>'))
                else:
                    language1Prob *= 0.007 #Default small probability if the conditional is not in the bigram
                if ((test_set[i][j], '</s>') in language2):
                    language2Prob *= language2.get((test_set[i][j], '</s>'))
                else:
                    language2Prob *= 0.007 #Default small probability if the conditional is not in the bigram
        if (language1Prob > language2Prob):  #If the first language has the HIGHER (OR SAME) probability of producing the word
            resultArray.append("language 1")
        elif (language1Prob == language2Prob):
            resultArray.append("language 1&2")  #If BOTH languages have EQUALLY likely chance of producing the word
        else:
            resultArray.append("language 2")    #If the first language has the LOWER probability of producing the word
    return resultArray #resultArray will be used to calculate accuracy

#The following function takes in 2 trigrams and a test set, and returns an array with the most probable language for each word in the test set
def compareTrigramAccuracy(language1, language2, test_set):
    resultArray = []  # Will hold the predicted language for each word (based on whichever bigram has highest probability)
    for i in range(0, len(test_set)):  # for each word in the array
        language1Prob = 1  # Will hold probability of word being from language 1
        language2Prob = 1  # Will hold probability of word being from language 2
        for j in range(0, len(test_set[i])):  # Loop through each character in word
            if(len(test_set[i]) == 1): #If a 1-letter word
                if(('<s>'+test_set[i][0], '</s>') in language1):
                    language1Prob *= language1.get(('<s>'+test_set[i][0], '</s>'))
                else:
                    language1Prob *= 0.01 #Default small probability if the conditional is not in the trigram
                if(('<s>'+test_set[i][0], '</s>') in language2):
                    language2Prob *= language2.get(('<s>'+test_set[i][0], '</s>'))
                else:
                    language2Prob *= 0.01 #Default small probability if the conditional is not in the trigram
            if(len(test_set[i]) != 1 and j > 1):  #General rules; If not at the second character of the word AND the word is longer than 1 character
                if((test_set[i][j-2]+test_set[i][j-1], test_set[i][j]) in language1):
                    language1Prob *= language1.get((test_set[i][j-2]+test_set[i][j-1], test_set[i][j]))
                else:
                    language1Prob *= 0.01 #Default small probability if the conditional is not in the trigram
                if ((test_set[i][j - 2] + test_set[i][j - 1], test_set[i][j]) in language2):
                    language2Prob *= language2.get((test_set[i][j - 2] + test_set[i][j - 1], test_set[i][j]))
                else:
                    language2Prob *= 0.01 #Default small probability if the conditional is not in the trigram
            elif(j == 1):   #At the second character of the word
                if(('<s>'+test_set[i][0], test_set[i][j]) in language1):
                    language1Prob *= language1.get(('<s>'+test_set[i][0], test_set[i][j]))
                else:
                    language1Prob *= 0.01 #Default small probability if the conditional is not in the trigram
                if (('<s>' + test_set[i][0], test_set[i][j]) in language2):
                    language2Prob *= language2.get(('<s>' + test_set[i][0], test_set[i][j]))
                else:
                    language2Prob *= 0.01 #Default small probability if the conditional is not in the trigram
            if(j == len(test_set[i]) - 1 and j != 0): #At the last character of the word (AND WORD IS LONGER THAN 1 LETTER)
                if((test_set[i][j-1]+test_set[i][j], '</s>') in language1):
                    language1Prob *= language1.get((test_set[i][j-1]+test_set[i][j], '</s>'))
                else:
                    language1Prob *= 0.01 #Default small probability if the conditional is not in the trigram
                if ((test_set[i][j - 1] + test_set[i][j], '</s>') in language2):
                    language2Prob *= language2.get((test_set[i][j - 1] + test_set[i][j], '</s>'))
                else:
                    language2Prob *= 0.01 #Default small probability if the conditional is not in the trigram
        if (language1Prob > language2Prob):  #If the first language has the HIGHER (OR SAME) probability of producing the word
            resultArray.append("language 1")
        elif(language1Prob == language2Prob):
            resultArray.append("language 1&2")  #If BOTH languages have EQUALLY likely chance of producing the word
        else:
            resultArray.append("language 2")    #If the first language has the LOWER probability of producing the word
    return resultArray #resultArray will be used to calculate accuracy

#The following function takes in an array from one of the 3 compareNgramAccuracy methods above, and returns the accuracy of the first N-gram predicting the words in the test set
def accuracyOfNgram(resultArray):
    count = 0 #Count of correct predicitons
    for i in range(0, len(resultArray)):
        if(resultArray[i] == "language 1"): #Correct prediction count goes up 1 if the first language is most likely for a word
            count += 1
        elif(resultArray[i] == "language 1&2"): #Correct prediction count goes up 1/2 if both languages are equally likely
            count += 0.5
    accuracy = count / len(resultArray) #Total count of correct predictions is divided by the length of the resultArray (length of the test set) to find the accuracy
    return accuracy #Returns the accuracy of the model as a double

#Training uni-, bi-, and tri-gram models for each language below
unigram_english, bigram_english, trigram_english = train(english_train)
unigram_french, bigram_french, trigram_french = train(french_train)
unigram_spanish, bigram_spanish, trigram_spanish = train(spanish_train)
unigram_italian, bigram_italian, trigram_italian = train(italian_train)


print("Accuracies of n-grams on test sets")
print("ENGLISH TEST SET")
accuracy_english1 = compareUnigramAccuracy(unigram_english, unigram_french, english_test) #Obtains array of which model had higher prediction per word in test set
accuracy_english2 = compareBigramAccuracy(bigram_english, bigram_french, english_test)
accuracy_english3 = compareTrigramAccuracy(trigram_english, trigram_french, english_test)
accuracy_english1 = accuracyOfNgram(accuracy_english1) #Obtains accuracy from array obtained above
accuracy_english2 = accuracyOfNgram(accuracy_english2)
accuracy_english3 = accuracyOfNgram(accuracy_english3)
print("Accuracy of English Unigram: ", accuracy_english1)
print("Accuracy of English Bigram: ", accuracy_english2)
print("Accuracy of English Trigram: ", accuracy_english3)

print("FRENCH TEST SET")
accuracy_french1 = compareUnigramAccuracy(unigram_french, unigram_english, french_test) #Obtains array of which model had higher prediction per word in test set
accuracy_french2 = compareBigramAccuracy(bigram_french, bigram_english, french_test)
accuracy_french3 = compareTrigramAccuracy(trigram_french, trigram_english, french_test)
accuracy_french1 = accuracyOfNgram(accuracy_french1) #Obtains accuracy from array obtained above
accuracy_french2 = accuracyOfNgram(accuracy_french2)
accuracy_french3 = accuracyOfNgram(accuracy_french3)
print("Accuracy of French Unigram: ", accuracy_french1)
print("Accuracy of French Bigram: ", accuracy_french2)
print("Accuracy of French Trigram: ", accuracy_french3)

print("SPANISH TEST SET")
accuracy_spanish1 = compareUnigramAccuracy(unigram_spanish, unigram_italian, spanish_test) #Obtains array of which model had higher prediction per word in test set
accuracy_spanish2 = compareBigramAccuracy(bigram_spanish, bigram_italian, spanish_test)
accuracy_spanish3 = compareTrigramAccuracy(trigram_spanish, trigram_italian, spanish_test)
accuracy_spanish1 = accuracyOfNgram(accuracy_spanish1) #Obtains accuracy from array obtained above
accuracy_spanish2 = accuracyOfNgram(accuracy_spanish2)
accuracy_spanish3 = accuracyOfNgram(accuracy_spanish3)
print("Accuracy of Spanish Unigram: ", accuracy_spanish1)
print("Accuracy of Spanish Bigram: ", accuracy_spanish2)
print("Accuracy of Spanish Trigram: ", accuracy_spanish3)

print("ITALIAN TEST SET")
accuracy_italian1 = compareUnigramAccuracy(unigram_italian, unigram_spanish, italian_test) #Obtains array of which model had higher prediction per word in test set
accuracy_italian2 = compareBigramAccuracy(bigram_italian, bigram_spanish, italian_test)
accuracy_italian3 = compareTrigramAccuracy(trigram_italian, trigram_spanish, italian_test)
accuracy_italian1 = accuracyOfNgram(accuracy_italian1) #Obtains accuracy from array obtained above
accuracy_italian2 = accuracyOfNgram(accuracy_italian2)
accuracy_italian3 = accuracyOfNgram(accuracy_italian3)
print("Accuracy of Italian Unigram: ", accuracy_italian1)
print("Accuracy of Italian Bigram: ", accuracy_italian2)
print("Accuracy of Italian Trigram: ", accuracy_italian3)


print("\nAccuracies of n-grams on dev sets")
print("ENGLISH DEV SET")
accuracy_english1 = compareUnigramAccuracy(unigram_english, unigram_french, english_dev) #Obtains array of which model had higher prediction per word in dev set
accuracy_english2 = compareBigramAccuracy(bigram_english, bigram_french, english_dev)
accuracy_english3 = compareTrigramAccuracy(trigram_english, trigram_french, english_dev)
accuracy_english1 = accuracyOfNgram(accuracy_english1) #Obtains accuracy from array obtained above
accuracy_english2 = accuracyOfNgram(accuracy_english2)
accuracy_english3 = accuracyOfNgram(accuracy_english3)
print("Accuracy of English Unigram: ", accuracy_english1)
print("Accuracy of English Bigram: ", accuracy_english2)
print("Accuracy of English Trigram: ", accuracy_english3)

print("FRENCH DEV SET")
accuracy_french1 = compareUnigramAccuracy(unigram_french, unigram_english, french_dev) #Obtains array of which model had higher prediction per word in dev set
accuracy_french2 = compareBigramAccuracy(bigram_french, bigram_english, french_dev)
accuracy_french3 = compareTrigramAccuracy(trigram_french, trigram_english, french_dev)
accuracy_french1 = accuracyOfNgram(accuracy_french1) #Obtains accuracy from array obtained above
accuracy_french2 = accuracyOfNgram(accuracy_french2)
accuracy_french3 = accuracyOfNgram(accuracy_french3)
print("Accuracy of French Unigram: ", accuracy_french1)
print("Accuracy of French Bigram: ", accuracy_french2)
print("Accuracy of French Trigram: ", accuracy_french3)

print("SPANISH DEV SET")
accuracy_spanish1 = compareUnigramAccuracy(unigram_spanish, unigram_italian, spanish_dev) #Obtains array of which model had higher prediction per word in dev set
accuracy_spanish2 = compareBigramAccuracy(bigram_spanish, bigram_italian, spanish_dev)
accuracy_spanish3 = compareTrigramAccuracy(trigram_spanish, trigram_italian, spanish_dev)
accuracy_spanish1 = accuracyOfNgram(accuracy_spanish1) #Obtains accuracy from array obtained above
accuracy_spanish2 = accuracyOfNgram(accuracy_spanish2)
accuracy_spanish3 = accuracyOfNgram(accuracy_spanish3)
print("Accuracy of Spanish Unigram: ", accuracy_spanish1)
print("Accuracy of Spanish Bigram: ", accuracy_spanish2)
print("Accuracy of Spanish Trigram: ", accuracy_spanish3)

print("ITALIAN DEV SET")
accuracy_italian1 = compareUnigramAccuracy(unigram_italian, unigram_spanish, italian_dev) #Obtains array of which model had higher prediction per word in dev set
accuracy_italian2 = compareBigramAccuracy(bigram_italian, bigram_spanish, italian_dev)
accuracy_italian3 = compareTrigramAccuracy(trigram_italian, trigram_spanish, italian_dev)
accuracy_italian1 = accuracyOfNgram(accuracy_italian1) #Obtains accuracy from array obtained above
accuracy_italian2 = accuracyOfNgram(accuracy_italian2)
accuracy_italian3 = accuracyOfNgram(accuracy_italian3)
print("Accuracy of Italian Unigram: ", accuracy_italian1)
print("Accuracy of Italian Bigram: ", accuracy_italian2)
print("Accuracy of Italian Trigram: ", accuracy_italian3)

#Prints N-grams below
print("\nN-grams")
print("English Unigram: ", unigram_english)
print("English Bigram: ",bigram_english)
print("English Trigram: ", trigram_english)

print("French Unigram: ", unigram_french)
print("French Bigram: ", bigram_french)
print("French Trigram: ", trigram_french)

print("Spanish Unigram: ", unigram_spanish)
print("Spanish Bigram: ", bigram_spanish)
print("Spanish Trigram: ", trigram_spanish)

print("Italian Unigram: ", unigram_italian)
print("Italian Bigram: ", bigram_italian)
print("Italian Trigram: ", trigram_italian)
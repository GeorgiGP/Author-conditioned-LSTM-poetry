#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2024/2025
##########################################################################
###
### Домашно задание 3
###
#############################################################################

import random

corpusSplitString = '@\n'
maxPoemLength = 10000
symbolCountThreshold = 100
authorCountThreshold = 20

def splitSentCorpus(fullSentCorpus, testFraction = 0.1):
    random.seed(42)
    random.shuffle(fullSentCorpus)
    testCount = int(len(fullSentCorpus) * testFraction)
    testSentCorpus = fullSentCorpus[:testCount]
    trainSentCorpus = fullSentCorpus[testCount:]
    return testSentCorpus, trainSentCorpus

def getAlphabetAuthors(corpus):
    symbols={}
    authors={}
    for s in corpus:
        if len(s) > 0:
            n=s.find('\n')
            aut = s[:n]
            if aut in authors: authors[aut] += 1
            else: authors[aut] = 1
            poem = s[n+1:]
            for c in poem:
                if c in symbols: symbols[c] += 1
                else: symbols[c]=1
    return symbols, authors

def prepareData(corpusFileName, startChar, endChar, unkChar, padChar):
    file = open(corpusFileName,'r')
    poems = file.read().split(corpusSplitString)
    symbols, authors = getAlphabetAuthors(poems)
    
    assert startChar not in symbols and endChar not in symbols and unkChar not in symbols and padChar not in symbols
    charset = [startChar,endChar,unkChar,padChar] + [c for c in sorted(symbols) if symbols[c] > symbolCountThreshold]
    char2id = { c:i for i,c in enumerate(charset)}
    authset = [a for a in sorted(authors) if authors[a] > authorCountThreshold]
    auth2id = { a:i for i,a in enumerate(authset)}
    
    corpus = []
    for i,s in enumerate(poems):
        if len(s) > 0:
            n=s.find('\n')
            aut = s[:n]
            poem = s[n+1:]
            corpus.append( (aut,[startChar] + [ poem[i] for i in range(min(len(poem),maxPoemLength)) ] + [endChar]) )

    testCorpus, trainCorpus  = splitSentCorpus(corpus, testFraction = 0.01)
    print('Corpus loading completed.')
    return testCorpus, trainCorpus, char2id, auth2id

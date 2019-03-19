# -*- coding: utf-8 -*-

import pickle
from janome.tokenizer import Tokenizer
import codecs

with open('uttr.pickle', mode='rb') as f:
    uttrList = pickle.load(f)
with open('res.pickle', mode='rb') as f:
    resList = pickle.load(f)
    
uttrText = ""
resText = ""
t = Tokenizer()

for uttr in uttrList:
    tokens = t.tokenize(uttr)
    seq = ""
    for token in tokens:
        if seq == "":
            seq = token.surface
        else:
            seq = seq + '\t' + token.surface
    print seq.encode('utf-8')
    if uttrText == '':
        uttrText = seq
    else:
        uttrText = uttrText + '\n' + seq
        
for res in resList:
    tokens = t.tokenize(res)
    seq = ""
    for token in tokens:
        if seq == "":
            seq = token.surface
        else:
            seq = seq + '\t' + token.surface
    print seq.encode('utf-8')
    if resText == '':
        resText = seq
    else:
        resText = resText + '\n' + seq
        
f = codecs.open('uttrTweet.txt', 'w', 'utf-8')
f.write(uttrText)
f.close()

f = codecs.open('resTweet.txt', 'w', 'utf-8')
f.write(resText)
f.close()
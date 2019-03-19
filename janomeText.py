# -*- coding: utf-8 -*-

import codecs
from janome.tokenizer import Tokenizer

f = codecs.open('replySource.txt', 'r', 'utf-8')
text = f.read()
text = text.split('\n')
t = Tokenizer()
newText = ''

for words in text:
    tokens = t.tokenize(words)
    seq = ''
    for token in tokens:
        if seq == '':
            seq = token.surface
        else:
            seq = seq + '\t' + token.surface
    if newText == '':
        newText = seq
    else:
        newText = newText + '\n' + seq
        
f = codecs.open('replySource2.txt', 'w', 'utf-8')
f.write(newText)
f.close()

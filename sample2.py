#!/usr/bin/env python                                                                                                                                                    
# -*- coding: utf-8 -*-                                                                                                                                                  

import sys
import numpy as np
from janome.tokenizer import Tokenizer
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import codecs



class seq2seq(chainer.Chain):
    def __init__(self, jv, ev, k, jvocab, evocab):
        super(seq2seq, self).__init__(
            embedx = L.EmbedID(jv, k),
            embedy = L.EmbedID(ev, k),
            H1 = L.LSTM(k, k),
            H2 = L.LSTM(k, k),
            H3 = L.LSTM(k, k),
            W = L.Linear(k, ev),
            )

    def __call__(self, jline, eline, jvocab, evocab):
        for i in range(len(jline)):
            wid = jvocab[jline[i]]
            x_k = self.embedx(Variable(np.array([wid], dtype=np.int32)))
            h1 = self.H1(x_k)
            h2 = self.H2(h1)
            h3 = self.H3(h2)
        x_k = self.embedx(Variable(np.array([jvocab['<eos>']], dtype=np.int32)))
        tx = Variable(np.array([evocab[eline[0]]], dtype=np.int32))
        h1 = self.H1(x_k)
        h2 = self.H2(h1)
        h3 = self.H3(h2)
        accum_loss = F.softmax_cross_entropy(self.W(h3), tx)
        for i in range(len(eline)):
            wid = evocab[eline[i]]
            x_k = self.embedy(Variable(np.array([wid], dtype=np.int32)))
            next_wid = evocab['<eos>'] if (i == len(eline) - 1) else evocab[eline[i+1]]
            tx = Variable(np.array([next_wid], dtype=np.int32))
            h1 = self.H1(x_k)
            h2 = self.H2(h1)
            h3 = self.H3(h2)
            loss = F.softmax_cross_entropy(self.W(h3), tx)
            accum_loss += loss

        return accum_loss

def mt(model, jline, id2wd, jvocab, evocab):
    for i in range(len(jline)):
        wid = jvocab[jline[i]]
        x_k = model.embedx(Variable(np.array([wid], dtype=np.int32), volatile='on'))
        h1 = model.H1(x_k)
        h2 = model.H2(h1)
        h3 = model.H3(h2)
    x_k = model.embedx(Variable(np.array([jvocab['<eos>']], dtype=np.int32), volatile='on'))
    h1 = model.H1(x_k)
    h2 = model.H2(h1)
    h3 = model.H3(h2)
    wid = np.argmax(F.softmax(model.W(h3)).data[0])
    if wid in id2wd:
        print id2wd[wid],
    else:
        print wid,
    loop = 0
    while (wid != evocab['<eos>']) and (loop <= 30):
        x_k = model.embedy(Variable(np.array([wid], dtype=np.int32), volatile='on'))
        h1 = model.H1(x_k)
        h2 = model.H2(h1)
        h3 = model.H3(h2)
        wid = np.argmax(F.softmax(model.W(h3)).data[0])
        if wid in id2wd:
            print id2wd[wid],
        else:
            print wid,
        loop += 1
    print

def constructVocabs(corpus, mod):

    vocab = {}
    id2wd = {}
    lines = codecs.open(corpus, 'rb', 'utf-8').read().split('\n')
    for i in range(len(lines)):
        lt = lines[i].split()
        for w in lt:
            if w not in vocab:
                if mod == "U":
                    vocab[w] = len(vocab)
                elif mod == "R":
                    id2wd[len(vocab)] = w
                    vocab[w] = len(vocab)

    if mod == "U":
        vocab['<eos>'] = len(vocab)
        v = len(vocab)
        return vocab, v
    elif mod == "R":
        id2wd[len(vocab)] = '<eos>'
        vocab['<eos>'] = len(vocab)
        v = len(vocab)
        return vocab, v, id2wd


argvs = sys.argv

_usage = """--                                                                                                                                                       
Usage:                                                                                                                                                                   
python generating.py [model] [uttranceDB] [responseDB]                                                                                                               
Args:                                                                                                                                                                    
[model]: The argument is seq2seq model to be trained using dialog corpus.                                                                                            
[utteranceDB]: The argument is utterance corpus to gain the distributed representation of words.                                                                     
[responseDB]: The argument is response corpus to gain the distributed representation of words.                                                                       
""".rstrip()

if len(argvs) < 4:
    print _usage
    sys.exit(0)


mpath = argvs[1]
utt_file = argvs[2]
res_file = argvs[3]

jvocab, jv = constructVocabs(utt_file, mod="U")
evocab, ev, id2wd = constructVocabs(res_file, mod="R")

demb = 256
model = seq2seq(jv, ev, demb, jvocab, evocab)
serializers.load_npz(mpath, model)
t = Tokenizer()
"""
while True:
    #utterance = raw_input()
    utterance = u"こんにちは"
    if utterance == "exit":
        print "Bye!!"
        sys.exit(0)
    tokens = t.tokenize(utterance)
    seq = []
    for token in tokens:
        seq.append(token.surface)
    jln = seq
    #jlnr = jln[::-1]
    jlnr = jln
    mt(model, jlnr, id2wd, jvocab, evocab)
"""
#utterance = raw_input()
utterance = u"好きな食べ物は何？"
if utterance == "exit":
    print "Bye!!"
    sys.exit(0)
tokens = t.tokenize(utterance)
seq = []
for token in tokens:
    seq.append(token.surface)
jln = seq
#jlnr = jln[::-1]
jlnr = jln
mt(model, jlnr, id2wd, jvocab, evocab)
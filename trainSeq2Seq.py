#!/usr/bin/env python                                                                                                                                                    
# -*- coding: utf-8 -*-                                                                                                                                                  
"""
対話学習用モデル学習スクリプト
uttr:発話に対してのres:返答を学習させる。
今のところ自然会話コーパスをプレトレーニングとして学習させ、チューニングとしてツイッター上のツイート、リプライのセットを学習させている。
"""

import sys
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
import codecs
import seq2seqChainer

# 初期値設定
pre_training_epochs = 100                         # pre-training学習回数
epochs = 10000                                    # チューニング学習回数
demb = 512                                        # 隠れ層ノード数
save_step = 1                                      # モデルを保存する頻度
utt_file_pre = "replySource2.txt"                 # pre-training用uttr
res_file_pre = "reply2.txt"                       # pre-training用res
utt_file_fin = "replySource2.txt"                 # チューニング用uttr
res_file_fin = "reply2.txt"                       # チューニング用res
out_path = "C:/Users/fujita.FILESERVER2/workspacePy/seq2seq-master/6_7_tweet_train_with_corpus_512_layer"            # 学習model出力先ディレクトリ

# 語彙の辞書作成
jvocab = {}
jlinesPre = codecs.open(utt_file_pre, 'rb', 'utf-8').read().split('\n')
for i in range(len(jlinesPre)):
    lt = jlinesPre[i].split()
    for w in lt:
        if w not in jvocab:
            jvocab[w] = len(jvocab)
jlinesFin = codecs.open(utt_file_fin, 'rb', 'utf-8').read().split('\n')
for i in range(len(jlinesFin)):
    lt = jlinesFin[i].split()
    for w in lt:
        if w not in jvocab:
            jvocab[w] = len(jvocab)

jvocab['<eos>'] = len(jvocab)
jv = len(jvocab)

evocab = {}
elinesPre = codecs.open(res_file_pre, 'rb', 'utf-8').read().split('\n')
for i in range(len(elinesPre)):
    lt = elinesPre[i].split()
    for w in lt:
        if w not in evocab:
            evocab[w] = len(evocab)
elinesFin = codecs.open(res_file_fin, 'rb', 'utf-8').read().split('\n')
for i in range(len(elinesFin)):
    lt = elinesFin[i].split()
    for w in lt:
        if w not in evocab:
            evocab[w] = len(evocab)
evocab['<eos>'] = len(evocab)
ev = len(evocab)

# model作成
model = seq2seqChainer(jv, ev, demb, jvocab, evocab)
# 最適化手法選択
optimizer = optimizers.Adam()
optimizer.setup(model)
# pre-training
for epoch in range(pre_training_epochs):
    sum_loss = 0
    # 1文ずつループ
    for i in range(len(jlinesPre)-1):
        jln = jlinesPre[i].split()
        jlnr = jln[::-1]
        eln = elinesPre[i].split()
        model.resetstate()
        model.zerograds()
        if len(eln) == 0:
            continue
        if len(jlnr) == 0:
            continue
        # 順伝搬の誤差を取得
        loss = model(jlnr, eln, jvocab, evocab)
        sum_loss += loss
        # 誤差を逆伝搬
        loss.backward()
        loss.unchain_backward()
        optimizer.update()
        if (i%100 == 0):
            print 'epoch:'
            print epoch
            print 'seq:'
            print i
    print 'epoch:'
    print epoch
    print 'sum_loss:'
    print sum_loss.data
# ファインチューニング
for epoch in range(epochs):
    sum_loss = 0
    for i in range(len(jlinesFin)-1):
        jln = jlinesFin[i].split()
        jlnr = jln[::-1]
        eln = elinesFin[i].split()
        model.H1.reset_state()
        model.H2.reset_state()
        model.H3.reset_state()
        model.zerograds()
        if len(eln) == 0:
            continue
        if len(jlnr) == 0:
            continue
        loss = model(jlnr, eln, jvocab, evocab)
        sum_loss += loss
        loss.backward()
        loss.unchain_backward()
        optimizer.update()
        if (i%100 == 0):
            print 'epoch:'
            print epoch
            print 'seq:'
            print i
    # save_stepごとにモデルを保存する
    if epoch != 0 and epoch%save_step == 0:
        outfile = out_path + "/seq2seq-" + str(epoch) + ".model"
        serializers.save_npz(outfile, model)
    print 'epoch:'
    print epoch
    print 'sum_loss:'
    print sum_loss.data
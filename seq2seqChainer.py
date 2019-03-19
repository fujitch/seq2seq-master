# -*- coding: utf-8 -*-
"""
対話学習用モデルクラス
param
jv:インプットの語彙数
ev:アウトプットの語彙数
k:隠れ層のノード数
jvocab:入力の語彙の辞書
evocab:出力の語彙の辞書
"""

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
import chainer.functions as F
import chainer.links as L


class seq2seqChainer(chainer.Chain):
    # initialize
    def __init__(self, jv, ev, k, jvocab, evocab):
        super(seq2seqChainer, self).__init__(
            embedx = L.EmbedID(jv, k),
            embedy = L.EmbedID(ev, k),
            H1 = L.LSTM(k, k),
            H2 = L.LSTM(k, k),
            H3 = L.LSTM(k, k),
            W = L.Linear(k, ev),
            )

    # 順伝搬関数
    def __call__(self, jline, eline, jvocab, evocab, dropout_ratio=0.5):
        # uttr(発話)を順伝搬させ、LSTMに記憶させる
        for i in range(len(jline)):
            wid = jvocab[jline[i]]
            x_k = self.embedx(Variable(np.array([wid], dtype=np.int32)))
            h1 = self.H1(F.dropout(x_k, ratio=dropout_ratio, train=True))
            h2 = self.H2(F.dropout(h1, ratio=dropout_ratio, train=True))
            h3 = self.H3(F.dropout(h2, ratio=dropout_ratio, train=True))
        # 予測スコアを出させて誤差を足していく
        x_k = self.embedx(Variable(np.array([jvocab['<eos>']], dtype=np.int32)))
        tx = Variable(np.array([evocab[eline[0]]], dtype=np.int32))
        h1 = self.H1(F.dropout(x_k, ratio=dropout_ratio, train=True))
        h2 = self.H2(F.dropout(h1, ratio=dropout_ratio, train=True))
        h3 = self.H3(F.dropout(h2, ratio=dropout_ratio, train=True))
        accum_loss = F.softmax_cross_entropy(self.W(h3), tx)
        for i in range(len(eline)):
            wid = evocab[eline[i]]
            x_k = self.embedy(Variable(np.array([wid], dtype=np.int32)))
            next_wid = evocab['<eos>'] if (i == len(eline) - 1) else evocab[eline[i+1]]
            tx = Variable(np.array([next_wid], dtype=np.int32))
            h1 = self.H1(F.dropout(x_k, ratio=dropout_ratio, train=True))
            h2 = self.H2(F.dropout(h1, ratio=dropout_ratio, train=True))
            h3 = self.H3(F.dropout(h2, ratio=dropout_ratio, train=True))
            loss = F.softmax_cross_entropy(self.W(h3), tx)
            accum_loss += loss
        return accum_loss
    # LSTM記憶領域削除
    def resetState(self):
        self.H1.reset_state()
        self.H2.reset_state()
        self.H3.reset_state()
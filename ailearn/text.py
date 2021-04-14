# Copyright 2018 Zhao Xingyu & An Yuexuan. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -*- coding: utf-8 -*-
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# 本文列表转化成index列表
def texts_to_sequences(texts, num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ',
                       char_level=False, oov_token=None, document_count=0, is_padding=True, maxlen=None, dtype='int32',
                       padding='pre', truncating='pre', value=0):
    '''
    :param texts: 待转换的文本
    :param num_words: None或整数，处理的最大单词数量。若被设置为整数，则分词器将被限制为待处理数据集中最常见的num_words个单词
    :param filters: 需要滤除的字符的列表或连接形成的字符串
    :param lower: 布尔值，是否将序列设为小写形式
    :param split: 字符串，单词的分隔符，如空格
    :param char_level: 如果为 True, 每个字符将被视为一个标记
    :param oov_token: out-of-vocabulary，如果给定一个string作为这个oov token的话，就将这个string也加到word_index，也就是从word到index 的映射中，用来代替那些字典上没有的字。
    :param document_count: 整数。分词器被训练的文档（文本或者序列）数量。仅在调用fit_on_texts或fit_on_sequences之后设置。
    :param is_padding: 布尔值，是否对序列进行填充
    :param maxlen: 序列的最大长度
    :param dtype: numpy数组的数据类型
    :param padding: 'pre'或'post'，在前面或后面填充
    :param truncating: 'pre'或'post'，在起始或结尾截断
    :param value: 填充值
    :return: 转化后的index序列，index到word的字典，word到index的字典
    '''
    tokenizer = Tokenizer(num_words=num_words, filters=filters, lower=lower, split=split, char_level=char_level,
                          oov_token=oov_token, document_count=document_count)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    index2word = tokenizer.index_word
    word2index = tokenizer.word_index
    if is_padding:
        sequences = pad_sequences(sequences, maxlen=maxlen, dtype=dtype, padding=padding, truncating=truncating,
                                  value=value)
    return sequences, index2word, word2index

# !/usr/bin/env python3
# -*- coding utf-8 -*-
# @TIME： 2020/10/23   20:21
# @FILE： classify_weibo.py
# @IDE ： PyCharm
# @contact: 980226547@qq.com
import joblib
import yaml
import pickle
import re
import numpy as np
from keras.models import model_from_yaml
from gensim.corpora.dictionary import Dictionary
from gensim.models.word2vec import Word2Vec
from keras.preprocessing import sequence
import pynlpir
import jieba


def svc_classify(content_path: str, topic: str):
    """
    初步分类广告
    :return:
    """
    pynlpir.open()
    stop = [line.strip() for line in open('.\\Sentiment_Analysis_ML\\SVM\\svm_data\\stop.txt', 'r', encoding='utf-8').readlines()]  # 停用词

    def read_file(filename):
        """
        读取文件，删除URL
        :param filename:
        :return:
        """
        f = open(filename, 'r', encoding='utf-8')
        line = f.readline()
        str = []
        while line:
            s = line
            p = re.compile(r'http?://.+$')  # 正则表达式，提取URL
            result = p.findall(line)  # 找出所有url
            if len(result):
                for i in result:
                    s = s.replace(i, '')  # 一个一个的删除
            temp = pynlpir.segment(s, pos_tagging=False)  # 分词
            for i in temp:
                if '@' in i:
                    temp.remove(i)  # 删除分词中的名字
            str.append(list(set(temp) - set(stop) - set('\u200b') - set(' ') - set('\u3000')))
            line = f.readline()
        return str

    def build_features(path: str):
        """
        为文本打标签：广告文本和正常文本
        :return:
        """
        feature = pickle.load(open("Sentiment_Analysis_ML/SVM/pynlpir_feature", "rb"))  # pynlpir获得300个特征词
        normalFeatures = []
        for items in read_file(path):
            a = {}
            for item in items:
                if item in feature.keys():  # 如果当前分词是特征词，则添加标签 true
                    a[item] = 'True'
            normalFeatures.append([a, "normal"])
        return normalFeatures
    # 分类
    print("load svc model...")
    svc_model = joblib.load(".\\Sentiment_Analysis_ML\\SVM\\svm.model")
    print("build features")
    content_features = build_features(content_path)
    print("pred")
    content_features, _ = zip(*content_features)
    pred = svc_model.classify_many(content_features)

    # 分类结果存储
    print("store results")
    file = open(content_path, "r", encoding="utf-8")
    write_file_adv = open(".\\Weibo_Data\\adv_data\\" + topic + ".txt", "w", encoding="utf-8")
    write_file_normal = open(".\\Weibo_Data\\normal_data\\" + topic + ".txt", "w", encoding="utf-8")
    file_content = file.readlines()
    for i in range(len(file_content)):
        if pred[i] == "adv":
            write_file_adv.write(file_content[i])
        else:
            write_file_normal.write(file_content[i])


def bayes_classify(content_path:str, topic: str):
    """
    贝叶斯分类
    :return:
    """
    stop = [line.strip() for line in open('.\\Sentiment_Analysis_ML\\SVM\\svm_data\\stop.txt', 'r', encoding='utf-8').readlines()]  # 停用词

    def loadDataSet(path):  # 返回每条微博的分词与标签
        line_cut = []
        with open(path, encoding="utf-8") as fp:
            for line in fp:
                temp = line.strip()
                try:
                    sentence = temp.lstrip()  # 每条微博
                    word_list = []
                    sentence = str(sentence).replace('\u200b', '')
                    for word in jieba.cut(sentence.strip()):
                        p = re.compile(r'\w', re.L)
                        result = p.sub("", word)
                        if not result or result == ' ':  # 空字符
                            continue
                        word_list.append(word)
                    word_list = list(set(word_list) - set(stop) - set('\u200b')
                                     - set(' ') - set('\u3000') - set('️'))
                    line_cut.append(word_list)
                except Exception:
                    continue
        return line_cut  # 返回每条微博的分词和标注

    def setOfWordsToVecTor(vocabularyList, moodWords):  # 每条微博向量化
        vocabMarked = [0] * len(vocabularyList)
        for smsWord in moodWords:
            if smsWord in vocabularyList:
                vocabMarked[vocabularyList.index(smsWord)] += 1
        return np.array(vocabMarked)

    def setOfWordsListToVecTor(vocabularyList, train_mood_array):  # 将所有微博准备向量化
        vocabMarkedList = []
        for i in range(len(train_mood_array)):
            vocabMarked = setOfWordsToVecTor(vocabularyList, train_mood_array[i])
            vocabMarkedList.append(vocabMarked)
        return vocabMarkedList

    print("load vocabList")
    vocab_file = open(".\\Sentiment_Analysis_ML\\bayes_vocablist.pkl", "rb")
    vocabList = pickle.load(vocab_file)
    print("line_cut")
    line_cut = loadDataSet(content_path)
    print("word list to vector")
    test_word_array = setOfWordsListToVecTor(vocabList, line_cut)
    print("load model")
    bayes_model = joblib.load("Sentiment_Analysis_ML\\bayes_gnb.model")
    print("predict")
    result = bayes_model.predict(test_word_array)
    print(result)

    pos_file = open(".\\Weibo_Data\\normal_data\\" + topic + "_pos.txt", "w", encoding="utf-8")
    neg_file = open(".\\Weibo_Data\\normal_data\\" + topic + "_neg.txt", "w", encoding="utf-8")
    file_content = open(content_path, "r", encoding="utf-8").readlines()
    for i in range(len(file_content)):
        if result[i] == 1:
            pos_file.write(file_content[i])
        elif result[i] == 0:
            neg_file.write(file_content[i])


def lstm_classify(content_path: str, topic: str):
    """
    lstm分类
    :return:
    """
    def create_dictionaries(model=None, combined=None):
        """
        创建词与索引的映射
        创建词与词向量的映射
        转换训练与测试的词语字典
        """
        maxlen = 200
        if (combined is not None) and (model is not None):
            gensim_dict = Dictionary()
            gensim_dict.doc2bow(model.wv.vocab, allow_update=True)
            w2indx = {v: k + 1 for k, v in gensim_dict.items()}  # 所有频数超过10的词语的索引
            w2vec = {word: model[word] for word in w2indx.keys()}  # 所有频数超过10的词语的词向量

            def parse_dataset(combined):
                """
                单词转整型数字
                """
                data = []
                for sentence in combined:
                    new_txt = []
                    for word in sentence:
                        try:
                            new_txt.append(w2indx[word])
                        except:
                            new_txt.append(0)
                    data.append(new_txt)
                return data

            combined = parse_dataset(combined)
            combined = sequence.pad_sequences(combined, maxlen=maxlen)
            # 每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
            return w2indx, w2vec, combined
        else:
            print('No data provided...')

    def input_transform(string_list):
        combine_list = []
        model = Word2Vec.load('./Sentiment_Analysis_LSTM/lstm_data/wiki.zh.text.model')
        num = 0
        for string in string_list:
            words = jieba.lcut(string)
            words = np.array(words).reshape(1, -1)
            _, _, combined = create_dictionaries(model, words)
            combined.reshape(1, -1)

            combine_list.append(combined)

            num += 1
            print(num)
            if num % 10000 == 0:
                print(num)
                break
        return combine_list

    print('loading model......')
    with open("./Sentiment_Analysis_LSTM/lstm_data/lstm.yml", 'r') as f:
        yaml_string = yaml.load(f)
    lstm_model = model_from_yaml(yaml_string)
    print('loading weights......')
    lstm_model.load_weights('./Sentiment_Analysis_LSTM/lstm_data/lstm.h5')
    lstm_model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    print('word to vec......')
    result_list = []
    file_content = open(content_path, "r", encoding="utf-8").readlines()
    string_list = [string.split("\n")[0] for string in file_content]
    data_list = input_transform(string_list)
    print("pred...")
    for data in data_list:
        result = lstm_model.predict_classes(data)
        print(result)
        if result[0][0] == 1:
            result_list.append(1)
        else:
            result_list.append(0)

    pos_file = open("./Weibo_Data/normal_data/" + topic + "_lstm_pos.txt", "w", encoding="utf-8")
    neg_file = open("./Weibo_Data/normal_data/" + topic + "_lstm_neg.txt", "w", encoding="utf-8")
    for i in range(10000):
        if result_list[i] == 1:
            pos_file.write(file_content[i])
        else:
            neg_file.write(file_content[i])


if __name__ == '__main__':
    svc_classify(content_path="./Weibo_Data/origin_data/trump_content.txt", topic="trump")
    # bayes_classify("./Weibo_Data/normal_data/disease.txt", topic="disease")
    # lstm_classify("./Weibo_Data/normal_data/disease.txt", topic="disease")




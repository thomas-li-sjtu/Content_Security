'''
用于情感分析初步清理数据形成content.pkl
所需的content.json是全关键词的微博内容
content.pkl形式为：
[
  [created_at, content原文, [content分词], keyword, weibo_url, like_num, repost_num, comment_num, tool]
]
'''

import codecs
import json
import pandas as pd
import jieba
import pickle
import re
from langconv import *

def Traditional2Simplified(sentence):
    '''
    将sentence中的繁体字转为简体字
    :param sentence: 待转换的句子
    :return: 将句子中繁体字转换为简体字之后的句子
    '''
    sentence = Converter('zh-hans').convert(sentence)
    return sentence


def Sent2Word(sentence):
    """Turn a sentence into tokenized word list and remove stop-word

    Using jieba to tokenize Chinese.

    Args:
        sentence: A string.

    Returns:
        words: A tokenized word list.
    """
    global stop_words

    words = jieba.cut(sentence)
    words = [w for w in words if w not in stop_words]

    return words


def Prepro(content, keyword):
    content_comment = []
    advertisement = ["王者荣耀", "券后", "售价", '¥', "￥", '下单']

    for k in range(0, len(content)):
        judge = []
        print('Processing train ', k)
        content[k]['content'] = Traditional2Simplified(content[k]['content'])
        # for adv in advertisement:
        #     if adv in content[k]['content']:
        #         judge.append("True")
        #         break
        # if re.search(r"买.*赠.*", content[k]['content']):
        #     judge.append("True")
        #     continue
        '''         
            [
              [created_at, content原文, [content分词], keyword, weibo_url, like_num, repost_num, comment_num, tool]  
            ]
        '''
        if "True" not in judge:
            comment_list = []
            comment_list.append(content[k]['created_at'])
            # comment_list.append(content[k]['content'])
            a_list = [
                re.compile(r'#.*?#'), re.compile(r'\[组图共.*张\]'), re.compile(r'http:.*'),
                re.compile(r'@.*? '), re.compile(r'\[.*?\]'), re.compile(r'【.*?】'), 
                re.compile(r'转发理由.*'), re.compile(r' .?.?.?.?.?.?.?.? 显示地图'),
                re.compile(u'[\U00010000-\U0010ffff]')
            ]
            for a in a_list:
                content[k]['content'] = a.sub('', content[k]['content'])
            comment_list.append(content[k]['content'])
            comment_list.append(Sent2Word(content[k]['content']))
            comment_list.append(keyword)
            url = content[k]['weibo_url']
            comment_list.append(url)
            try:
                comment_list.append(content[k]['like_num'])
            except:
                comment_list.append(' ')
            try:
                comment_list.append(content[k]['repost_num'])
            except:
                comment_list.append(' ')
            try:
                comment_list.append(content[k]['comment_num'])
            except:
                comment_list.append(' ')

            try:
                comment_list.append(content[k]['tool'])
            except:
                comment_list.append(' ')
            content_comment.append(comment_list)

    pickle.dump(content_comment, open(f'./result/{keyword}.pkl', 'wb'))


if __name__ == '__main__':
    print("停用词读取")
    stop_words = [w.strip() for w in open('./dict/哈工大停用词表.txt', 'r', encoding='UTF-8').readlines()]
    stop_words.extend(['\n', '\t', ' ', '回复', '转发微博', '转发', '微博', '秒拍', '秒拍视频', '视频', "王者荣耀", "王者", "荣耀"])
    for i in range(128000, 128722 + 1):
        stop_words.extend(chr(i))
    # print(stop_words)

    print("content读取")
    keyword = 'disease'
    with open(f'./data/{keyword}.json','r',encoding='utf8')as fp:
        json_data = json.load(fp)
        # print(len(json_data))
        # print(json_data[0])

    Prepro(json_data, keyword)

    with open(f'./result/{keyword}.pkl', 'rb') as f:
        content = pickle.load(f)

    result = []
    for con in content:
        result.append(con[1].strip())

    pickle.dump(list(set(result)), open(f'./result/{keyword}-content.pkl', 'wb'))

import wordcloud
import pickle


def gen_wordcloud(topic: str, path: str):
    """
    生成词云  基于内容的词云
    """
    content = open(path, "r", encoding="utf-8").read()
    stop = [line.strip() for line in open('F:\PythonCode\PycharmCode\Content_Security\Sentiment_Analysis_ML\SVM\svm_data\\stop.txt', 'r', encoding='utf-8').readlines()]
    w = wordcloud.WordCloud(width=1000, font_path='./msyh.ttc', background_color='white', height=700,
                            stopwords=set(stop))
    w.generate(content.strip())
    w.to_file(topic + '_contert.png')


if __name__ == '__main__':
    gen_wordcloud("trump", '..\\Weibo_Data\\normal_data\\trump.txt')

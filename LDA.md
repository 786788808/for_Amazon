## LDA主题分析
###### 在做市场调查的时候，需要了解竞品情况，知道对手的卖点、缺点。能在选品、产品迭代时，给开发、运营提供一定的建议、方向。
###### 如何快速了解市场产品情况呢？
###### 于是，决定先对产品评论reviews下手，从消费者的评论中，用LDA主题分析来找到关键的几个点，再结合开发、供应商的资料，给出看法。  
思路：
- 首先，爬取相关产品的 reviews,并且分类：【1-3】星作为差评，【4-5】星作为好评。
- 然后， 用 nltk 和 re 库，进行分词、去除标点符号、去除停用词、词形还原等操作。
- 最后，就是用 Gensim 来完成最后的主题分析。

具体操作：  
1.获取数据部分：这里用插件 `Instant Data Scraper`,爬取亚马逊美国站的评论数据。  
2.初步整理爬虫数据：每款产品爬取的结果列都会有出入，所以保留星评、评论标题及评论主题。合并产品标题及评论主题。  
3.根据星评级数，将好评positive、差评negative分别放入txt中：positive.txt、negative.txt。  
4.清洗、分词等，然后做出词云图，最后做LDA主题分析。
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Hush
# @FileName: marketing_LDA_model_function.py
# @Software: PyCharm


import matplotlib.pyplot as plt
import re 
import imageio  # 调入词云图背景图 mask
from wordcloud import WordCloud  # 词云图
# from nltk import word_tokenize
from nltk.corpus import stopwords  # 停用词
from nltk.stem import WordNetLemmatizer  # 词形还原
from autocorrect import Speller  # 单词拼写检查
from gensim import corpora, models  # LDA主题分析用

# 定义基本路径：词云背景图、好评内容、好评词云图、差评内容、差评词云图路径
word_cloud_mask_location = r'E:\报表合计\市调\Aquarium light\dd.jpg'
positive_reviews_location = r'E:\报表合计\市调\Aquarium light\positive.txt'
positive_wordcloud_location = r'E:\报表合计\市调\Aquarium light\positive_cloud.png'
negative_reviews_location = r'E:\报表合计\市调\Aquarium light\negative.txt'
negative_wordcloud_location = r'E:\报表合计\市调\Aquarium light\negative_cloud.png'


def lda_function():
'''
手动输入好评或差评，接着输入主题个量，即可获得相应的主题数
'''
    a = input("要分析好评还是差评？请输入 好评 或者 差评 二字：\n")
    if a == '好评':
        print('——————现在开始好评分析——————：\n')
        # 读文件
        with open(positive_reviews_location, 'r', encoding='utf-8-sig') as f:
            positive_review = f.read()
        # 小写处理
        out_words = positive_review.lower()
        # 拓展缩写
        out_words = re.sub(r"doesn('|’)t", "does not", out_words)
        out_words = re.sub(r"doesnt", "does not", out_words)
        out_words = re.sub("aren('|’)t", "are not", out_words)
        out_words = re.sub("arnt", "are not", out_words)
        out_words = re.sub("don('|’)t", "do not", out_words)
        out_words = re.sub("dont", "do not", out_words)
        out_words = re.sub("can('|’)t", "can not", out_words)
        out_words = re.sub("didn('|’)t", "did not", out_words)
        out_words = re.sub("didnt", "did not", out_words)
        out_words = re.sub("couldn('|’)t", "could not", out_words)
        out_words = re.sub("couldnt", "could not", out_words)
        out_words = re.sub("i('|’)ve", "i have", out_words)
        out_words = re.sub("i('|’)m", "i am", out_words)
        out_words = re.sub("^im$", "i am", out_words)
        out_words = re.sub("i('|’)ll", "i will", out_words)
        out_words = re.sub("i('|’)d", "i would", out_words)
        out_words = re.sub("it('|’)s", "it is", out_words)
        out_words = re.sub("isn('|’)t", "is not", out_words)
        out_words = re.sub("ive", "i have", out_words)
        out_words = re.sub('havent', "have not", out_words)
        out_words = re.sub('hadnt', "had not", out_words)
        out_words = re.sub("wouldn('|’)t", "would not", out_words)
        out_words = re.sub("wouldnt", "would not", out_words)
        out_words = re.sub("wasn('|’)t", "was not", out_words)
        out_words = re.sub("wasnt", "was not", out_words)
        out_words = re.sub("werent", "were not", out_words)
        out_words = re.sub("won('|’)t", "will not", out_words)
        out_words = re.sub("whats", "what is", out_words)
        out_words = re.sub("you('|’)ll", "you will", out_words)
        out_words = re.sub("youre", "you are", out_words)
        out_words = re.sub("youd", "you would", out_words)
        out_words = re.sub("there('|’)s", "there is", out_words)
        out_words = re.sub("thats", "that is", out_words)
        out_words = re.sub("that('|’)s", "that is", out_words)
        out_words = re.sub("theyre", "they are", out_words)
        out_words = re.sub('theyve', "they have", out_words)
        out_words = re.sub("gotta", "have got to", out_words)
        out_words = re.sub("whats", "what is", out_words)
        out_words = out_words.replace("hdi", "hdmi")
        out_words = out_words.replace('cause', 'because')
        out_words = out_words.replace('kinda', 'kind of')
        }
        # 拼写纠错，暂时用这个，还有 pyenchant，暂时不用
        correct = Speller()
        out_words = correct.autocorrect_sentence(out_words)
        # 分词，没有直接用 word_tokenize,因为没有完全分开，直接用 split 根据多种符号去分开
        out_words = re.split("\.| |\\n|\,|\!|\/|\-|\:|\=", out_words)
        print('分词后共有词数：\n', len(out_words))
        out_words_correct_hdmi = []
        # 修改'hdi'为'hdmi'（为爬取一款HDMI线时用到，其余产品需根据实际情况更改）
        for i in out_words:
            if i == 'hdi':
                i = 'hdmi'
                out_words_correct_hdmi.append(i)
            else:
                out_words_correct_hdmi.append(i)
                
        # 去除标点符号、多余字符、数字
        out_words_without_symbol = []
        for word in out_words_correct_hdmi:
            no_symbol = re.sub("[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+", "", word)
            no_num = re.sub('[0-9]', "", no_symbol)
            out_words_without_symbol.append(no_num)
        out_words = [x for x in out_words_without_symbol if x != ""]
        # 去除前后多余的空格
        out_words = [x.strip() for x in out_words]
        # 去除停用词
        stop_words = stopwords.words('english')
        out_words = [word for word in out_words if word not in stop_words]
        # 词形还原
        wl = WordNetLemmatizer()
        out_words = [wl.lemmatize(word) for word in out_words]
        # TODO 修改emoji部分，修改re.sub部分
        # 去除 emoji 表情
        def delete_emoji(original_text, new_string=''):
            try:
                emoji_str = re.compile(u'[\U00010000-\U0010ffff]')
            except re.error:
                emoji_str = re.compile(u'[\uD800-\uDBFF][\uDC00-\uDFFF]')
            return emoji_str.sub(new_string, original_text)

        out_words_without_emoji = []
        for i in out_words:
            word_without_emoji = delete_emoji(i, new_string='')
            out_words_without_emoji.append(word_without_emoji)
        print(out_words_without_emoji)

        out_words = [x for x in out_words_without_emoji if len(x) > 2]
        print('去掉停用词后剩下的词数：\n', len(out_words))
        print('整理后的词：\n', out_words)
        # 做词云图
        mk = imageio.imread(word_cloud_mask_location)  # 读取图片
        wc = WordCloud(background_color='white',  # 设置背景色
                       max_words=100,  # 设置显示多少词
                       font_path='msyh.ttc',  # 设置字体
                       collocations=False,  # 避免重复词
                       mask=mk,  # 设置图形形状
                       contour_width=1.5,  # 设置图形边缘线条宽度
                       contour_color='steelblue'  # 设置边缘线颜色
                       ).generate(positive_review)
        plt.imshow(wc)
        plt.axis("off")  # 设置无坐标轴
        plt.show()
        wc.to_file(positive_wordcloud_location)
        pos_dict = corpora.Dictionary([out_words])  # 建立词典
        print('建立词典:\n', pos_dict)
        pos_corpus = [pos_dict.doc2bow([i]) for i in out_words]  # 建立语料库，bag of word
        print('建立词袋：\n', pos_corpus)
        num_of_topics = int(input("请输入LDA主题数量，直接输入数字：\n"))
        neg_lda = models.LdaModel(pos_corpus, num_topics=num_of_topics, id2word=pos_dict)  # LDA模型训练
        for i in range(num_of_topics):
            print('—————— 第%d个 positive_topic——————' % (int(i)+1))
            print(neg_lda.print_topic(i))  # 输出第i个 LDA主题

    else:
        print('——————现在开始差评分析——————：\n')
        # 读文件
        with open(negative_reviews_location, 'r', encoding='utf-8-sig') as f:
            negative_review = f.read()
        # 小写处理
        out_words = negative_review.lower()
        # 拓展缩写
        out_words = re.sub(r"doesn('|’)t", "does not", out_words)
        out_words = re.sub(r"doesnt", "does not", out_words)
        out_words = re.sub("aren('|’)t", "are not", out_words)
        out_words = re.sub("arnt", "are not", out_words)
        out_words = re.sub("don('|’)t", "do not", out_words)
        out_words = re.sub("dont", "do not", out_words)
        out_words = re.sub("can('|’)t", "can not", out_words)
        out_words = re.sub("didn('|’)t", "did not", out_words)
        out_words = re.sub("didnt", "did not", out_words)
        out_words = re.sub("couldn('|’)t", "could not", out_words)
        out_words = re.sub("couldnt", "could not", out_words)
        out_words = re.sub("i('|’)ve", "i have", out_words)
        out_words = re.sub("i('|’)m", "i am", out_words)
        out_words = re.sub("^im$", "i am", out_words)
        out_words = re.sub("i('|’)ll", "i will", out_words)
        out_words = re.sub("i('|’)d", "i would", out_words)
        out_words = re.sub("it('|’)s", "it is", out_words)
        out_words = re.sub("isn('|’)t", "is not", out_words)
        out_words = re.sub("ive", "i have", out_words)
        out_words = re.sub('havent', "have not", out_words)
        out_words = re.sub('hadnt', "had not", out_words)
        out_words = re.sub("wouldn('|’)t", "would not", out_words)
        out_words = re.sub("wouldnt", "would not", out_words)
        out_words = re.sub("wasn('|’)t", "was not", out_words)
        out_words = re.sub("wasnt", "was not", out_words)
        out_words = re.sub("werent", "were not", out_words)
        out_words = re.sub("won('|’)t", "will not", out_words)
        out_words = re.sub("whats", "what is", out_words)
        out_words = re.sub("you('|’)ll", "you will", out_words)
        out_words = re.sub("youre", "you are", out_words)
        out_words = re.sub("youd", "you would", out_words)
        out_words = re.sub("there('|’)s", "there is", out_words)
        out_words = re.sub("thats", "that is", out_words)
        out_words = re.sub("that('|’)s", "that is", out_words)
        out_words = re.sub("theyre", "they are", out_words)
        out_words = re.sub('theyve', "they have", out_words)
        out_words = re.sub("gotta", "have got to", out_words)
        out_words = re.sub("whats", "what is", out_words)
        out_words = out_words.replace("hdi", "hdmi")
        out_words = out_words.replace('cause', 'because')
        out_words = out_words.replace('kinda', 'kind of')
        # 拼写纠错，暂时用这个，还有 pyenchant，暂时不用
        correct = Speller()
        out_words = correct.autocorrect_sentence(out_words)
        # 分词，没有直接用 word_tokenize,因为没有完全分开，直接用 split 根据多种符号去分开
        out_words = re.split("\.| |\\n|\,|\!|\/|\-|\:|\=", out_words)
        print('分词后共有词数：\n', len(out_words))
        out_words_correct_hdmi = []
        # 修改'hdi'为'hdmi'
        for i in out_words:
            if i == 'hdi':
                i = 'hdmi'
                out_words_correct_hdmi.append(i)
            else:
                out_words_correct_hdmi.append(i)

        # 去除标点符号、多余字符、数字
        out_words_without_symbol = []
        for word in out_words_correct_hdmi:
            no_symbol = re.sub("[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+", "", word)
            no_num = re.sub('[0-9]', "", no_symbol)
            out_words_without_symbol.append(no_num)
        out_words = [x for x in out_words_without_symbol if x != ""]
        # 去除前后多余的空格
        out_words = [x.strip() for x in out_words]
        # 去除停用词
        stop_words = stopwords.words('english')
        out_words = [word for word in out_words if word not in stop_words]
        # 词形还原
        wl = WordNetLemmatizer()
        out_words = [wl.lemmatize(word) for word in out_words]
        # 去除 emoji 表情
        def delete_emoji(original_text, new_string=''):
            try:
                emoji_str = re.compile(u'[\U00010000-\U0010ffff]')
            except re.error:
                emoji_str = re.compile(u'[\uD800-\uDBFF][\uDC00-\uDFFF]')
            return emoji_str.sub(new_string, original_text)

        out_words_without_emoji = []
        for i in out_words:
            word_without_emoji = delete_emoji(i, new_string='')
            out_words_without_emoji.append(word_without_emoji)
        print(out_words_without_emoji)

        out_words = [x for x in out_words_without_emoji if len(x) > 2]
        print('去掉停用词后剩下的词数：\n', len(out_words))
        print('整理后的词：\n', out_words)
        # 做词云图
        mk = imageio.imread(word_cloud_mask_location)  # 读取图片
        wc = WordCloud(background_color='white',  # 设置背景色
                       max_words=100,  # 设置显示多少词
                       font_path='msyh.ttc',  # 设置字体
                       collocations=False,  # 避免重复词
                       mask=mk,  # 设置图形形状
                       contour_width=1.5,  # 设置图形边缘线条宽度
                       contour_color='steelblue'  # 设置边缘线颜色
                       ).generate(negative_review)
        plt.imshow(wc)
        plt.axis("off")  # 设置无坐标轴
        plt.show()
        wc.to_file(negative_wordcloud_location)
        neg_dict = corpora.Dictionary([out_words])  # 建立词典
        print('建立词典:\n', neg_dict)
        neg_corpus = [neg_dict.doc2bow([i]) for i in out_words]  # 建立语料库，bag of word
        print('建立词袋：\n', neg_corpus)
        num_of_topics = int(input("请输入LDA主题数量，直接输入数字：\n"))
        neg_lda = models.LdaModel(neg_corpus, num_topics=num_of_topics, id2word=neg_dict)  # LDA模型训练
        for i in range(num_of_topics):
            print('—————— 第%d个 negative_topic——————' % (int(i) + 1))
            print(neg_lda.print_topic(i))  # 输出第i个 LDA主题

    print('分析结束')


if __name__ == '__main__':
    lda_function()
    
```

##### 反思：
整个分词的过程有待优化，参考其他大神的文章后，依然用自己的思路逐步清除下来。  
拓展缩写部分，需修改，当前部分太累赘。  
去除emoji表情部分，仍不是很懂，怕有遗漏，后面要优化。  
整体代码太长，处理那一part可以直接换成通用函数，直接调用即可，不用写两遍。



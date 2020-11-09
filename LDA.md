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
3.根据星评级数，将好评positive、差评negative分别放入txt中。  
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Hush
# @FileName: marketing_LDA_model_function.py
# @Software: PyCharm


import matplotlib.pyplot as plt
import re  # 分词用到
import imageio  # 调入词云图背景图 mask
from wordcloud import WordCloud  # 词云图
# from nltk import word_tokenize
from nltk.corpus import stopwords  # 停用词
from nltk.stem import WordNetLemmatizer  # 词形还原
from autocorrect import Speller  # 单词拼写检查
from gensim import corpora, models  # LDA主题分析

word_cloud_mask_location = r'E:\报表合计\市调\Aquarium light\dd.jpg'
positive_reviews_location = r'E:\报表合计\市调\Aquarium light\positive.txt'
positive_wordcloud_location = r'E:\报表合计\市调\Aquarium light\positive_cloud.png'
negative_reviews_location = r'E:\报表合计\市调\Aquarium light\negative.txt'
negative_wordcloud_location = r'E:\报表合计\市调\Aquarium light\negative_cloud.png'
```





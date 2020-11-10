##### 背景：
##### 想要研究客户的特征，从而来提升店铺销售额，但是亚马逊并没有提供客户的个人信息（个人隐私），也没有用户的行为数据，所以客户从什么渠道来的、页面逗留时间、加购情况等等，一概不知。但是，基本的交易数据还是能拿到的。客户的邮箱（作为ID鉴别）、交易时间、交易金额都有，可以做个RFM客户价值分析，看看店铺里的客户情况是怎么样的。再根据该情况，给运营、推广提供建议。

###### RFM可以不用python来写，excel都可以完成。但是写个脚本，以后就可以一键操作啦
该模型采用经典的二分法，将客户分成经典的八块。划分点可用中位数、平均数、二八划分值，具体怎么分，看各自的店铺情况。


```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Hush
# @FileName: RMF_20190915.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import datetime
import pymysql
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_rows', None)  # 显示全部行
pd.set_option('display.max_columns', None)  # 显示全部列
plt.rcParams['font.sans-serif'] = ['SimHei']   # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False     # 正常显示负号

# 连接数据库数据
conn = pymysql.connect(
                    host='localhost',
                    port=3306,
                    user='root',
                    password='******',  # 数据库密码
                    db='xin_***',  # 数据库名
                    charset='utf8')
cursor = conn.cursor()
# 修改下单日期，从datetime形式为date形式
sql_1 = 'update rmf_20190915 set `下单日期_orders`=left(`下单日期_orders`,10)'
cursor.execute(sql_1)
sql_2 = 'select * from rmf_20190915'
df_original = pd.read_sql(sql_2, con=conn)
conn.close()
# 提取所需的属性，其余撇掉
df = df_original[['付款人邮箱', '下单日期_orders', 'amazon-order-id_orders', 'product sales_Order', '店铺_orders']]
# 查看数据的缺失情况
print("------原始订单数据：------\n", df.info())
print("------数值型数据情况：------\n", df.describe())
print("-----需要的数据里是否有缺失值：\n", df.isnull().any(axis=0))
# sns.violinplot(y="product sales_Order", data=df, palette="Set3")
sns.violinplot(x="店铺_orders", y="product sales_Order", data=df, palette="Set3")
plt.title("各店铺销售额分布情况", fontsize="xx-large")
plt.xlabel("店铺")
plt.ylabel("销售额（付款）")
plt.show()


# 将各店铺数据区分开来
shop = ['US1', 'US2', 'US11', 'US14', 'US15', 'UK11', 'UK14']
df_US1 = df[df['店铺_orders'] == 'US1']
df_US2 = df[df['店铺_orders'] == 'US2']
df_US11 = df[df['店铺_orders'] == 'US11']
df_US14 = df[df['店铺_orders'] == 'US14']
df_US15 = df[df['店铺_orders'] == 'US15']
df_UK11 = df[df['店铺_orders'] == 'UK11']
df_UK14 = df[df['店铺_orders'] == 'UK14']
shop_df = [df_US1, df_US2, df_US11, df_US14, df_US15, df_UK11, df_UK14]
print('店铺分批完毕')
# 数据转换

print('每个店铺客户的价值分析开始：\n')
writer = pd.ExcelWriter(r'E:\报表合计\RMF\RMF_20190915\RMF_20190915_result.xlsx')
for store in shop_df:
    last_date_consumer = store['下单日期_orders'].groupby(store['付款人邮箱']).max()
    final_date = datetime.datetime(2019, 9, 15)  # 指定一个时间节点，用于计算其他时间与该时间的距离
    Recency = (final_date - last_date_consumer).dt.days  # 计算R间隔
    Monetary = store['product sales_Order'].groupby(store['付款人邮箱']).sum()
    Frequency = store['下单日期_orders'].groupby(store['付款人邮箱']).count()

    RMF_data = [Recency, Monetary, Frequency]  # 将r、m、f三个维度组成列表
    RMF_col = ['Recency', 'Monetary', 'Frequency']  # 设置r、m、f三个维度列名
    RMF_df = pd.DataFrame(np.array(RMF_data).transpose(), dtype='int32', columns=RMF_col, index=Frequency.index)
    RMF_df['category'] = 0
    # print(RMF_df.head(5))
    if RMF_df.empty:
        print('————该店铺没有数据————\n')
    else:
        # print('检查是否有缺失值：\n',RMF_df.isnull().any())
        # Rmf三个维度的均值
        store_name = store['店铺_orders'].values
        store_name = store_name[0]
        print('————%s店铺情况:————' % store_name)
        ave_r = RMF_df['Recency'].median()
        ave_m = RMF_df['Monetary'].median()
        ave_f = RMF_df['Frequency'].median()
        print(RMF_df.describe())
        print(ave_r, ave_m, ave_f)

        print(RMF_df.shape)
        i = 0
        max_col_num = int(RMF_df.shape[0])
        while i <= max_col_num:
            if (RMF_df.iloc[i, 0] < ave_r) & (RMF_df.iloc[i, 1] >= ave_m) & (RMF_df.iloc[i, 2] >= ave_f):
                RMF_df.iloc[i, 3] = 1
            elif (RMF_df.iloc[i, 0] < ave_r) & (RMF_df.iloc[i, 1] >= ave_m) & (RMF_df.iloc[i, 2] < ave_f):
                RMF_df.iloc[i, 3] = 2
            elif (RMF_df.iloc[i, 0] >= ave_r) & (RMF_df.iloc[i, 1] >= ave_m) & (RMF_df.iloc[i, 2] >= ave_f):
                RMF_df.iloc[i, 3] = 3
            elif (RMF_df.iloc[i, 0] >= ave_r) & (RMF_df.iloc[i, 1] >= ave_m) & (RMF_df.iloc[i, 2] < ave_f):
                RMF_df.iloc[i, 3] = 4
            elif (RMF_df.iloc[i, 0] < ave_r) & (RMF_df.iloc[i, 1] < ave_m) & (RMF_df.iloc[i, 2] >= ave_f):
                RMF_df.iloc[i, 3] = 5
            elif (RMF_df.iloc[i, 0] < ave_r) & (RMF_df.iloc[i, 1] < ave_m) & (RMF_df.iloc[i, 2] < ave_f):
                RMF_df.iloc[i, 3] = 6
            elif (RMF_df.iloc[i, 0] >= ave_r) & (RMF_df.iloc[i, 1] < ave_m) & (RMF_df.iloc[i, 2] >= ave_f):
                RMF_df.iloc[i, 3] = 7
            else:
                RMF_df.iloc[i, 3] = 8

            i += 1
            if i == max_col_num:
                break

        RMF_df['店铺'] = store_name
        print(RMF_df.tail(2))
        RMF_df.to_excel(writer, sheet_name='%s' % store_name)
        # writer.save()
        print('%s店铺结束!!!!!!!!' % store_name)
writer.save()


print('分组完毕~')
```

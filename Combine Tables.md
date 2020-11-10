##### 手上有多家亚马逊店铺，平时下载多店铺数据已经要花费不少时间，而每个店铺数据的列常常是不一样的，如果要整理合并统计这些数据，单靠excel又得消耗很长时间。  
##### 为了减少整理报表的时间、降低出错率，于是决定写个脚本，实现快速合并多个店铺的数据：合并主要的、有用的列，没用的列直接撇掉不要了。
手上负责的是美国、英国及澳洲的店铺，但即使是同一站点的报表，数据列也可能不一样，而且亚马逊的报表格式也会随着时间改变。需要根据自己的店铺情况改动。  
order表是动态变化的，但是只要有下单记录，就会有订单号记录，其余的交易数量、交易金额等是动态变化的，无法追踪，想要知道下单数量、下单金额，不可实现。  
工作中主要看付款金额，即每日流水Payment里的情况，客人付款、退款都是分开行记录的，于是可根据订单号情况，记录该订单付款、退款、亚马逊物流损坏等情况。  

```python
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: Hush
# @Software: PyCharm


import os
import time
import pandas as pd
import numpy as np
import chardet
from glob import glob


# 1.整理报表：所有订单 order 的函数：111
def concat_order():
    """
    :return:返回所有订单数据
    """
    print('——————开始聚合所有订单的报表数据——————')
    order = pd.DataFrame()
    order_path_head = r'C:\Users\Administrator\Desktop\report\order'
    order_list = os.listdir(order_path_head)
    num = 0  # 后面进行到哪张表

    for i in order_list:
        print('——开始处理%s表——' % (i))
        order_path = order_path_head + '\\' + i
        f = open(order_path, mode = 'rb')
        data = f.read()
        encod = chardet.detect(data)  # 返回一个关于编码信息的字典，这里只需要字典的值，不要键
        encod = list(encod.values())[0]
        order_new = pd.read_table(order_path, sep='\t', encoding=encod)
        order_new['店铺'] = i.split(sep='.')[0]  # 补充店铺列信息
        # 不同店铺的变量名稍有出入，有的会在头尾多空白符，需要去掉，避免数据concat后出现错误
        col_name_list = order_new.columns.tolist()
        order_new.columns = [i.strip() for i in col_name_list]
        order = pd.concat([order, order_new], ignore_index=True, sort=False)
        num = num + 1
        print('第%d张表：%s合并完毕\n' % (num, i))

    # 输出汇总的所有订单 order 数据，放在同一路径下
    order.to_excel(order_path_head + r'\order.xlsx', index=False)
    print("报告：已经输出结果，'所有订单 order'可以收割！！！")


# 2.整理报表： payment 报告的函数：111
def concat_payment():
    """
    :return: 返回31列，日期用字符串去表示了，后期最好改回struc_time格式或者正常的年月日 时分秒
    """
    print('——————开始聚合 payment 的报表数据——————')
    payment = pd.DataFrame()
    pay_path_head = r'C:\Users\Administrator\Desktop\report\payment'
    pay_list = os.listdir(pay_path_head)
    num = 0  # 后面进行到哪张表

    for i in pay_list:
        print('——开始处理%s表——' % (i))
        pay_path = pay_path_head + '\\' + i
        f = open(pay_path, mode = 'rb')
        data = f.read()
        encod = chardet.detect(data)  # 返回一个关于编码信息的字典，这里只需要字典的值，不要键
        encod = list(encod.values())[0]

        if i[0:2] == 'UK':  # 英国店铺开头有6行解释，美国和澳大利亚有7行解释语句
            pay_new = pd.read_csv(pay_path, encoding=encod, skiprows=6)
        else:
            pay_new = pd.read_csv(pay_path, encoding=encod, skiprows=7)

        pay_new['店铺'] = i.split(sep='.')[0]  # 补充店铺列信息
        # 不同店铺的变量名稍有出入，有的会在头尾多空白符，需要去掉，避免数据concat后出现错误
        col_name_list = pay_new.columns.tolist()
        ##  美国与英国的fulfillment fulfilment不同，统一用'fulfillment'
        col_name_list = ['fulfillment' if i == 'fulfilment' else i for i in col_name_list]
        pay_new.columns = [i.strip() for i in col_name_list]

        payment = pd.concat([payment, pay_new], ignore_index=True, sort=False)
        num = num + 1
        print('第%d张表：%s合并完毕\n' % (num, i))

    # 输出汇总的所有 payment 数据，放在同一路径下
    payment.to_excel(pay_path_head + r'\payment.xlsx', index=False)
    print("报告：已经输出结果，'payment报表'可以收割！！！")


# 3.整理报表： adv 报告的函数：111
def concat_adv():
    print('——————开始聚合广告的报表数据——————')
    adv = pd.DataFrame()
    adv_path_head = r'C:\Users\Administrator\Desktop\report\adv'
    adv_list = os.listdir(adv_path_head)
    num = 0  # 后面进行到哪张表

    for i in adv_list:
        print('——开始处理%s表——' % (i))
        adv_path = adv_path_head + '\\' + i
        f = open(adv_path, mode='rb')
        data = f.read()
        encod = chardet.detect(data)  # 返回一个关于编码信息的字典，这里只需要字典的值，不要键
        encod = list(encod.values())[0]
        print(encod)
        if encod in ('GB2312','gb2312'):
            print('need modify')
            encod = 'gb18030'
        else:
            print('no change')
            pass
        print('修改后的编码： ', encod)
        adv_new = pd.read_csv(adv_path, encoding=encod)
        adv_new['店铺'] = i.split(sep='.')[0]  # 补充店铺列信息
        # 部分列名头尾有空白符号，需要去掉
        col_name_list = adv_new.columns.tolist()
        adv_new.columns = [i.strip() for i in col_name_list]

        # 部分列存在$符号，或者%符，在数据库里已经定义为decimal，所以要去掉这两个符号，最后再变回float类型
        # 日期格式是字符串形式，要改
        ## （1）除去$符号
        money_list = ['花费', '预算', '每次点击成本(CPC)', '去年支出', '去年每次点击成本(CPC)', '7天总销售额(￥)']
        f_money = lambda x: float(str(x).replace('$', '').replace(',', '').replace('£', '').replace('￡',''))  # 先转成字符串将其中的符号去掉，再转回float
        for mon in money_list:  # 除掉$号
            adv_new[mon] = adv_new[mon].apply(f_money)
            adv_new[mon] = adv_new[mon].replace('nan','')  # 只写上一句的话会返回nan，暂时没有找原因，后面改此处

        ## （2）除去%符号，并且缩小100倍
        percent_list = ['点击率(CTR)', '广告成本销售比(ACoS)']
        f_percent = lambda x: float(x.replace('%', '')) / 100
        for per in percent_list:  # 除掉百分号并且缩小100倍
            adv_new[per] = adv_new[per].apply(f_percent)

        ## (3)日期数据（有3列,结束日期基本为空，不处理，剩下2列）呈现字符串形式，现在将其转换成数据库中用到的'2020-04-01'形式
        date_list = ['日期', '开始日期']

        def alter_date(x):
            if '-' in x:
                date_structed = time.mktime(time.strptime(x, '%d-%b-%y'))
                x = time.strftime('%Y-%m-%d', time.localtime(date_structed))
            else:
                date_structed = time.mktime(time.strptime(x, '%b %d, %Y'))
                x = time.strftime('%Y-%m-%d', time.localtime(date_structed))
            return x

        for da in date_list:
            adv_new[da] = adv_new[da].apply(alter_date)

        adv = pd.concat([adv, adv_new], ignore_index=True, sort=False)
        num = num + 1
        print('第%d张表：%s合并完毕\n' % (num, i))

    # 输出汇总的所有订单数据，放在同一路径下
    adv.to_excel(adv_path_head + r'\adv.xlsx', index=False)
    print("报告：已经输出结果，'广告报表'可以收割！！！") ##


# 4.整理周UV报表的函数：
def concat_week_uv():
    pass


# 5.整理月UV报表的函数：
def concat_month_uv():
    pass


# 6.整理换货订单报表的函数：
def concat_replaces():
    """
    :replaces:返回所有换货数据，只有美国有换货
    """
    print('——————开始聚合所有换货订单的报表数据——————')
    replaces = pd.DataFrame()
    replaces_path_head = r'C:\Users\Administrator\Desktop\report\replaces'
    replaces_list = os.listdir(replaces_path_head)
    num = 0  # 后面进行到哪张表

    for i in replaces_list:
        print('——开始处理%s表——' % (i))
        replaces_path = replaces_path_head + '\\' + i
        f = open(replaces_path, mode = 'rb')
        data = f.read()
        encod = chardet.detect(data)  # 返回一个关于编码信息的字典，这里只需要字典的值，不要键
        encod = list(encod.values())[0]
        replaces_new = pd.read_csv(replaces_path, encoding=encod)
        replaces_new['店铺'] = i.split(sep='.')[0]  # 补充店铺列信息
        # 不同店铺的变量名稍有出入，有的会在头尾多空白符，需要去掉，避免数据concat后出现错误
        col_name_list = replaces_new.columns.tolist()
        replaces_new.columns = [i.strip() for i in col_name_list]
        replaces = pd.concat([replaces, replaces_new], ignore_index=True, sort=False)
        num = num + 1
        print('第%d张表：%s合并完毕\n' % (num, i))

    # 输出汇总的所有订单 replaces 数据，放在同一路径下
    replaces.to_excel(replaces_path_head + r'\replaces.xlsx', index=False)
    print("报告：已经输出结果，'所有换货订单 replaces '可以收割！！！")


# 7.整理退货订单报表的函数：
def concat_returns():
    """
    :returns:返回所有退货数据
    """
    print('——————开始聚合所有退货订单的报表数据——————')
    returns = pd.DataFrame()
    returns_path_head = r'C:\Users\Administrator\Desktop\report\returns'
    returns_list = os.listdir(returns_path_head)
    num = 0  # 后面进行到哪张表

    for i in returns_list:
        print('——开始处理%s表——' % (i))
        returns_path = returns_path_head + '\\' + i
        f = open(returns_path, mode='rb')
        data = f.read()
        encod = chardet.detect(data)  # 返回一个关于编码信息的字典，这里只需要字典的值，不要键
        encod = list(encod.values())[0]
        returns_new = pd.read_csv(returns_path, encoding=encod)
        returns_new['店铺'] = i.split(sep='.')[0]  # 补充店铺列信息
        col_name_list = returns_new.columns.tolist()

        if i=='AU1.csv':
            original_col= [i.strip() for i in col_name_list]
            replace_dict = {'Returned Date': 'return-date', 'Order ID': 'order-id', 'Merchant SKU':'sku', 'ASIN':'asin',
                   'FNSKU':'fnsku', 'Title':'product-name', 'Quantity':'quantity', 'FC':'fulfillment-center-id',
                   'Disposition':'detailed-disposition', 'Reason':'reason', 'Status':'status',
                   'lpn':'license-plate-number', 'Customer comments':'customer-comments'}
            returns_new.columns = [replace_dict[i] if i in replace_dict else i for i in original_col]
        else:
            returns_new.columns = [i.strip() for i in col_name_list]

        returns = pd.concat([returns, returns_new], ignore_index=True, sort=False)
        num = num + 1
        print('第%d张表：%s合并完毕\n' % (num, i))

    # 输出汇总的所有订单 returns 数据，放在同一路径下
    returns.to_excel(returns_path_head + r'\returns.xlsx', index=False)
    print("报告：已经输出结果，'所有退货订单 returns '可以收割！！！")


# 8.整理每日UV报表的函数：
def concat_uv_day():
    """
    :return:返回每日uv数据
    """
    print('——————开始聚合每日UV的报表数据——————')
    uv_day = pd.DataFrame()
    uv_path_head = r'C:\Users\Administrator\Desktop\report\uv_day'
    uv_path_head = glob(uv_path_head + '\\*')  # 所有完整路径的子文件夹（在一个列表里）
    num = 0  # 后面进行到哪张表
    for j in uv_path_head:
        uv_day_list = os.listdir(j)
        # print(j)
        store = j.split('\\')[6]
        print(store)
        # store = j.
        for i in uv_day_list:
            print('——开始处理%s: %s表——' % (store, i))
            uv_day_path = j + '\\' + i
            f = open(uv_day_path, mode='rb')
            data = f.read()
            encod = chardet.detect(data)  # 返回一个关于编码信息的字典，这里只需要字典的值，不要键
            encod = list(encod.values())[0]
            # print(('——开始处理%s: %s表:编码:%s') % (store, i, encod))
            if encod=='GB2312':
                encod = 'gb18030'
            else:
                pass
            uv_day_new = pd.read_csv(uv_day_path, encoding=encod)
            uv_day_new['店铺'] = store  # 补充店铺列信息
            uv_day_new['日期'] = i.split('.')[0] # 补充哪一天的数据
            # 不同店铺的变量名稍有出入，有的会在头尾多空白符，需要去掉，避免数据concat后出现错误
            col_name_list = uv_day_new.columns.tolist()
            uv_day_new.columns = [i.strip() for i in col_name_list]

            # 将多余的符号去除：
            # （1）去除多余的货币符号
            money_list = ['已订购商品销售额', '已订购商品的销售额 – B2B']
            f_money = lambda x: str(x).replace('$', '').replace('AUD', '').replace('£', '').replace('US', '')
            for mon in money_list:
                try:
                    uv_day_new[mon] = uv_day_new[mon].apply(f_money)
                except:
                    pass

            # (2)去除多余的百分号，并且将其转化为float小数点
            percent_list = ['买家访问次数百分比', '页面浏览次数百分比', '购买按钮赢得率', '订单商品数量转化率', '商品转化率 – B2B']
            f_percent = lambda x: float(str(x).replace('%',''))/100
            for per in percent_list:
                try:
                    uv_day_new[per] = uv_day_new[per].apply(f_percent)
                except:
                    pass

            uv_day = pd.concat([uv_day, uv_day_new], ignore_index=True, sort=False)
            num = num + 1
            print('第%d张表：%s合并完毕\n' % (num, i))
        print('%s店铺合并完毕\n' % store)

    # 输出汇总的所有订单 uv_day 数据，放在同一路径下
    uv_day.to_excel(r'C:\Users\Administrator\Desktop\report\uv_day' + r'\uv_day.xlsx', index=False)
    print("报告：已经输出结果，'所有订单 uv_day '可以收割！！！")


# 9.整理每日库存历史记录函数：
def concat_inventory_day():
    """
    :return: 每日库存记录
    """
    print('——————开始聚合 inventory 的报表数据——————')
    inventory = pd.DataFrame()
    inventory_path_head = r'C:\Users\Administrator\Desktop\report\inventory'
    inventory_list = os.listdir(inventory_path_head)
    num = 0  # 后面进行到哪张表

    for i in inventory_list:
        print('——开始处理%s表——' % (i))
        inventory_path = inventory_path_head + '\\' + i
        f = open(inventory_path, mode = 'rb')
        data = f.read()
        encod = chardet.detect(data)  # 返回一个关于编码信息的字典，这里只需要字典的值，不要键
        encod = list(encod.values())[0]

        if i == 'AU1.csv':
            inventory_new = pd.read_csv(inventory_path, encoding='Latin-1')
        else:
            inventory_new = pd.read_csv(inventory_path, encoding=encod)

        inventory_new['店铺'] = i.split(sep='.')[0]  # 补充店铺列信息
        # 不同店铺的变量名稍有出入，有的会在头尾多空白符，需要去掉，避免数据concat后出现错误
        col_name_list = inventory_new.columns.tolist()
        inventory_new.columns = [i.strip() for i in col_name_list]

        inventory = pd.concat([inventory, inventory_new], ignore_index=True, sort=False)
        num = num + 1
        print('第%d张表：%s合并完毕\n' % (num, i))

    # 输出汇总的所有 inventory 数据，放在同一路径下
    inventory.to_excel(inventory_path_head + r'\inventory.xlsx', index=False)
    print("报告：已经输出结果，'inventory报表'可以收割！！！")


# 10.整理promotion记录函数：
def concat_promotion():
    """
    :return: 每日优惠券使用记录
    """
    print('——————开始聚合 promotion 的报表数据——————')
    promotion = pd.DataFrame()
    promotion_path_head = r'C:\Users\Administrator\Desktop\report\promotion'
    promotion_list = os.listdir(promotion_path_head)
    num = 0  # 后面进行到哪张表

    for i in promotion_list:
        print('——开始处理%s表——' % (i))
        promotion_path = promotion_path_head + '\\' + i
        f = open(promotion_path, mode = 'rb')
        data = f.read()
        encod = chardet.detect(data)  # 返回一个关于编码信息的字典，这里只需要字典的值，不要键
        encod = list(encod.values())[0]
        promotion_new = pd.read_csv(promotion_path, encoding=encod)

        promotion_new['店铺'] = i.split(sep='.')[0]  # 补充店铺列信息
        # 不同店铺的变量名稍有出入，有的会在头尾多空白符，需要去掉，避免数据concat后出现错误
        col_name_list = promotion_new.columns.tolist()
        promotion_new.columns = [i.strip() for i in col_name_list]

        promotion = pd.concat([promotion, promotion_new], ignore_index=True, sort=False)
        num = num + 1
        print('第%d张表：%s合并完毕\n' % (num, i))

    # 输出汇总的所有 promotion 数据，放在同一路径下
    promotion.to_excel(promotion_path_head + r'\promotion.xlsx', index=False)
    print("报告：已经输出结果，'promotion 报表'可以收割！！！")



# 11.最后综合上诉函数，写一个函数包含上面函数:
def concat_report():
    report = ['payment', 'order', 'adv', 'week_uv', 'month_uv', 'replaces', 'returns', 'uv_day', 'inventory', 'promotion']
    mission = input('请输入所需处理的报表类型:在payment, order, adv, week_uv, month_uv, replaces, returns, uv_day, inventory, promotion中选一个')

    if mission==report[0]:
        concat_payment()
    elif mission==report[1]:
        concat_order()
    elif mission==report[2]:
        concat_adv()
    elif mission==report[3]:
        concat_week_uv()
    elif mission==report[4]:
        concat_month_uv()
    elif mission==report[5]:
        concat_replaces()
    elif mission==report[6]:
        concat_returns()
    elif mission==report[7]:
        concat_uv_day()
    elif mission==report[8]:
        concat_inventory_day()
    else:
        concat_promotion()



if __name__=='__main__':
    concat_report()

```

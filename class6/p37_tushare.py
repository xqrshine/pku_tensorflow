import tensorflow as tf
import tushare as ts
import matplotlib.pyplot as plt


# 加载股票代码是600519的指定时间的数据
df1 = ts.get_k_data('600519', ktype='D', start='2010-10-30', end='2020-10-30')
datapath1 = ('./SH600519.csv')
# 将读取的数据保存为csv文件
df1.to_csv(datapath1)
print(df1)
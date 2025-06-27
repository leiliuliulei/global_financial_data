import pandas as pd
import numpy as np


file_dir = r'格力.xlsx'
df = pd.read_excel(file_dir, index_col=0, parse_dates=True)

# 计算K线
df = df.assign(diff_high=df.High.diff(), diff_low=df.Low.diff())

conditions = [(df.diff_high <= 0) & (df.diff_low <= 0), (df.diff_high >= 0) & (df.diff_low >= 0)]
df['k_line'] = np.select(condlist=conditions, choicelist=['下降', '上升'], default='None')
df = df.replace('None', None)
df['k_line'] = df['k_line'].ffill()

# 计算分型
pattern_list = df.k_line.tolist()

# 初始化处理位置
j = 0
while j < len(pattern_list) - 2:

    if pattern_list[j: j + 3] == ['下降', '下降', '上升']:
        pattern_list[j: j + 3] = ['底'] * 3  # 标记底分型
        j += 3  # 跳过这三行

    elif pattern_list[j: j + 3] == ['上升', '上升', '下降']:
        pattern_list[j: j + 3] = ['顶'] * 3  # 标记顶分型
        j += 3  # 跳过这三行

    else:
        j += 1  # 继续检查下一行

# 将结果转换回 pandas Series（如果需要）
df['pattern'] = pattern_list

df.to_excel(r'chan.xlsx')
# 计算分型
# df['k_line_prev'] = df['k_line'].shift(1)
# choices = ['下降k线', '底分型', '上升k线', '顶分型', '下降k线', '上升k线']
# conditions = [(df.k_line_prev == '下降') & (df.k_line == '下降'),
#               (df.k_line_prev == '下降') & (df.k_line == '上升'),
#               (df.k_line_prev == '上升') & (df.k_line == '上升'),
#               (df.k_line_prev == '上升') & (df.k_line == '下降'),
#               (df.k_line_prev is None) & (df.k_line == '下降'),
#               (df.k_line_prev is None) & (df.k_line == '上升')]
#
# df['分型'] = np.select(condlist=conditions, choicelist=choices, default='None')
# df = df.replace('None', None)

# df.to_excel(r'格力_diff.xlsx')


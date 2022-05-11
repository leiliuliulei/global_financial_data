import pandas as pd
import plotly_express as px


def industry_mapping(engine):

    sql_string = r'select F034V, F036V, F038V, SECNAME, SECCODE from stock2100 ' \
                 r'where SECCODE REGEXP "^[0-9]" and CHANGE_CODE <>2 and F034V is not null ' \
                 r'ORDER BY F034V, F036V, F038V, SECNAME;'

    all_company_df = pd.read_sql_query(sql_string, engine)

    all_stock_name = all_company_df.SECNAME.unique().tolist()   # 所有上市公司名称
    all_industry = all_company_df.F034V.unique().tolist()       # 申万一级行业列表

    # 下面的变量名：d代表dict，1、2、3代表申万一、二、三级行业，n代表name（上市公司名称）, c代表code（上市公司代码）
    d_12 = option_dicts(all_company_df, 'F034V', 'F036V')       # 特定申万一级行业包含的二级行业
    d_23 = option_dicts(all_company_df, 'F036V', 'F038V')       # 特定申万二级行业包含的三级行业
    d_3n = option_dicts(all_company_df, 'F038V', 'SECNAME')     # 特定申万三级行业包含的上市公司

    d_n1 = value_dicts(all_company_df, 'SECNAME', 'F034V')      # 特定上市公司对应的一级行业
    d_n2 = value_dicts(all_company_df, 'SECNAME', 'F036V')      # 特定上市公司对应的二级行业
    d_n3 = value_dicts(all_company_df, 'SECNAME', 'F038V')      # 特定上市公司对应的三级行业

    d_nc = all_company_df.set_index('SECNAME').SECCODE.to_dict()    # 上市公司名称对应的证券代码

    return all_stock_name, all_industry, d_12, d_23, d_3n, d_n1, d_n2, d_n3, d_nc


def option_dicts(all_company_df, key_column, value_column):
    raw_dict = all_company_df.groupby(key_column)[value_column].unique().to_dict()
    fine_dict = {k: v.tolist() for (k, v) in raw_dict.items()}
    return fine_dict


def value_dicts(all_company_df, key_column, value_column):
    raw_dict = all_company_df.groupby(key_column)[value_column].unique().to_dict()
    fine_dict = {k: str(v.squeeze()) for (k, v) in raw_dict.items()}
    return fine_dict


def rank_by_revenue(secname_list, engine):

    sql_string = r'select SECNAME from stock2301 where SECNAME in ({l}) and ' \
                 r'F001D = "2020-12-31" and ' \
                 r'F002V = "071001" and ' \
                 r'CHANGE_CODE <> 2 ' \
                 r'order by F006N desc;'.format(l=list_to_string(secname_list))

    result = pd.read_sql_query(sql_string, engine).squeeze().tolist()
    return result


def get_financial_data(secname_list, name_dict, engine):

    name_string = list_to_string(secname_list)
    name_criteria = 'where SECNAME in ({l}) '.format(l=name_string)

    head = 'select * from '
    end = 'and CHANGE_CODE <> 2;'

    sql_string_2300 = head + 'stock2300 ' + name_criteria + 'and F001D REGEXP "12-31" and F002V = "071001" ' + end
    sql_string_2301 = head + 'stock2301 ' + name_criteria + 'and F001D REGEXP "12-31" and F002V = "071001" ' + end
    sql_string_2303 = head + 'stock2303 ' + name_criteria + 'and F069D REGEXP "12-31" and F070V = "071001" ' + end

    df_2300 = pd.read_sql_query(sql_string_2300, engine).rename(columns=name_dict['stock2300'])
    df_2301 = pd.read_sql_query(sql_string_2301, engine).rename(columns=name_dict['stock2301'])
    df_2303 = pd.read_sql_query(sql_string_2303, engine).rename(columns=name_dict['stock2303'])

    df_2300 = reindex_by_year_and_name(df_2300, secname_list, 2021, 10)
    df_2301 = reindex_by_year_and_name(df_2301, secname_list, 2021, 10)
    df_2303 = reindex_by_year_and_name(df_2303, secname_list, 2021, 10)

    return df_2300, df_2301, df_2303


def reindex_by_year_and_name(original_df, secname_list, newest_year, number_of_year):
    # 这个函数按照报告年度、公司列表，对SQL查询到的df进行reindex。因为这样reindex之后得到的df是有固定的框架的，不会因为某年公司多或少导致数据不同

    year_list = range(newest_year, newest_year - number_of_year, -1)        # 从今年开始，倒数 number_of_year 年

    original_df['报告年度'] = pd.to_datetime(original_df['报告年度']).dt.year     # 从“报告年度”列中挑出“年”，数据类型是int64
    ordered_index = pd.MultiIndex.from_product([year_list, secname_list], names=['报告年度', '证券简称'])
    reindex_df = original_df.set_index(['报告年度', '证券简称']).reindex(index=ordered_index)

    return reindex_df


def list_to_string(a_list):
    list_with_quotation = ['\"' + a_item + '\"' for a_item in a_list if a_item]
    return r', '.join(list_with_quotation)


def translation_dictionary():
    # 数据库中不同的表格（如stock2300, stock2301），字段名称都需要翻译。这个函数读取字段名称翻译（一个excel文件），变成字典。
    # 字典的key是表格名（如stock2300, stock2301），value是这个表格专属的字典。

    translation_excel_path = 'name_dictionary.xlsx'
    sheet_list = ['stock2300', 'stock2301', 'stock2302', 'stock2303']
    sheet_dfs = pd.read_excel(translation_excel_path, sheet_name=sheet_list)

    sheet_dict = {k: v.set_index('英文名称')['中文名称'].to_dict() for (k, v) in sheet_dfs.items()}
    return sheet_dict


def pick_and_rename(raw_df, needed_columns, new_name_dict=None):
    if new_name_dict:
        return raw_df.rename(columns=new_name_dict)[needed_columns]     # 修改列名并只挑出需要的列
    else:
        return raw_df[needed_columns]


def is_single_company(df):
    if len(df['证券简称'].unique()) == 1:
        return True
    else:
        return False


def yi(df):
    return df.divide(1.e8).round(1)     # 这个函数返回以“亿”为单位的数据。1.e8是1亿。


def income_chart(raw_df):

    new_names = {'归属于母公司所有者的净利润': '归母净利润', '扣除非经常性损益后的净利润(2007版)': '扣非净利润', '经营活动现金流量净额': '经营现金流'}
    needed_col = ['营业收入', '营业利润', '利润总额', '净利润', '归母净利润', '扣非净利润', '经营现金流']

    df = pick_and_rename(raw_df, needed_col, new_names)
    # df = df.divide(pow(10, 8)).round(1)

    return bar_figure(yi(df), '收入对比（亿）', y_cols=needed_col)


def cost_chart(raw_df):

    new_names = {'一、营业总收入': '营业总收入', '其中：营业成本': '营业成本', '营业税金及附加': '营业税', '加：公允价值变动净收益': '公允价值变动收益'}

    revenue_col = ['营业总收入']
    subtract_col = ['财务费用', '研发费用', '管理费用', '销售费用', '营业税', '利息支出', '营业成本']
    supplement_col = ['投资收益', '公允价值变动收益', '信用减值损失（2019格式）', '资产减值损失（2019格式）', '资产减值损失', '资产处置收益', '其它收入']

    df = pick_and_rename(raw_df, revenue_col + subtract_col + supplement_col, new_names)

    # 这里跟一个squeeze()是为了把只有1个column的dataframe变成series，否则后面得到percentage_df那一步会全是Nan。
    revenue = df[revenue_col].squeeze()
    subtract_part = df[subtract_col]

    nominal_operating_profit = revenue - subtract_part.sum(axis=1)
    nominal_operating_profit.name = '名义营业利润'

    # 注意这里对补充部分（supplement_part）统一取负数。因为cost分析这里的各种cost(subtract_col那些列)在报表里都是正数，
    # 而supplement_part这些补充营业利润的项目在报表里也是正数。这里特意变成了负数以把他们这些项目体现在横轴之下，表现为对nominal营业利润的补充。
    supplement_part = - df[supplement_col]

    combined_df = pd.concat([nominal_operating_profit, subtract_part, supplement_part], axis=1)
    percentage_df = combined_df.divide(revenue, axis='index').round(2)
    needed_col = ['名义营业利润'] + subtract_col + supplement_col

    return bar_figure(percentage_df, '成本拆解', y_cols=needed_col, y_percent=True, stack=True)


def efficiency_chart(raw_df):
    new_names = {'总资产报酬率': 'ROA', '加权平均净资产收益率': '加权平均ROE'}
    needed_col = ['毛利率', '营业利润率', '净利润率', 'ROA', '加权平均ROE']

    df = pick_and_rename(raw_df, needed_col, new_names)
    return bar_figure(df, '运营效率对比', y_cols=needed_col)


def warren_touch_chart(raw_df, company_name):
    needed_col = ['营运资金', '营业收入', '营业利润']
    df = pick_and_rename(raw_df, needed_col).dropna(how='all')
    # df = df.xs(company_name, level='证券简称')
    return area_figure(df, r"Warren's touch:{c}".format(c=company_name), y_cols=needed_col)


def balance_sheet_chart(raw_df):

    current_asset_total = ['流动资产合计']
    current_asset_cols = ['货币资金', '交易性金融资产', '衍生金融资产', '一年内到期的非流动资产', '应收票据', '应收账款', '预付款项', '存货']

    fixed_asset_total = ['非流动资产合计']
    fixed_asset_cols = ['债权投资', '长期股权投资', '固定资产', '在建工程', '无形资产', '商誉', '递延所得税资产']

    needed_cols = current_asset_total + current_asset_cols + fixed_asset_total + fixed_asset_cols
    df = pick_and_rename(raw_df, needed_cols)

    df['其他流动资产'] = df[current_asset_total] - df[current_asset_cols].sum(axis=1)
    df['其他固定资产'] = df[fixed_asset_total] - df[fixed_asset_cols].sum(axis=1)

    needed_cols = current_asset_cols + ['其他流动资产'] + fixed_asset_cols + ['其他固定资产']
    return bar_figure(yi(df), '资产表', y_cols=needed_cols, stack=True)


def bar_figure(df, title, y_cols, stack=False, y_percent=False):

    df = df.fillna(0).reset_index()
    the_bar_mode = 'relative' if stack else 'group'

    if is_single_company(df):
        fig = px.bar(data_frame=df, x='报告年度', y=y_cols, title=title, opacity=0.8)
    else:
        fig = px.bar(data_frame=df, x='证券简称', y=y_cols, animation_frame='报告年度', title=title, opacity=0.8)

    fig.layout.update(xaxis_title='', yaxis_title='', title=dict(x=0.5), barmode=the_bar_mode)
    # fig.layout.update(xaxis_autorange='reversed')

    if y_percent:
        fig.layout.update(yaxis=dict(tickformat='.0%'))

    return fig


def area_figure(df, title, y_cols):

    df.reset_index(inplace=True)
    fig = px.area(data_frame=df, x='报告年度', y=y_cols, title=title)
    fig.layout.update(xaxis_title='', yaxis_title='', title=dict(x=0.5))

    return fig

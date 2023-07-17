import pandas as pd
import plotly.express as px
from datetime import datetime


class MySQLQuery(object):

    def __init__(self, engine, dictionary_file_path):

        self.__engine = engine

        sheet_list = ['stock2100', 'stock2300', 'stock2301', 'stock2302', 'stock2303', 'stock2401', 'stock2402']
        sheet_dfs = pd.read_excel(dictionary_file_path, sheet_name=sheet_list)
        database_dicts = {k: v.set_index('英文名称')['中文名称'].to_dict() for (k, v) in sheet_dfs.items()}

        self.__name_book = database_dicts

    def __query(self, query_string, table_name):
        df = pd.read_sql_query(query_string, self.__engine).rename(columns=self.__name_book[table_name])

        if '报告年度' in df.columns:
            df['报告年度'] = pd.to_datetime(df['报告年度'])
        elif '交易日期' in df.columns:
            df['交易日期'] = pd.to_datetime(df['交易日期'])

        return df

    def get_industry_info(self):

        query_industry = r'select F034V, F036V, F038V, SECNAME, SECCODE from stock2100 ' \
                         r'where SECCODE REGEXP "^[0-9]" and CHANGE_CODE <>2 and F034V is not null;'

        all_company_df = self.__query(query_industry, 'stock2100')

        all_stock_name = all_company_df['证券简称'].unique().tolist()           # 所有上市公司名称
        all_industry = all_company_df['申万行业分类一级名称'].unique().tolist()   # 申万一级行业列表

        # 下面的变量名：d代表dict，1、2、3代表申万一、二、三级行业，n代表name（上市公司名称）, c代表code（上市公司代码）
        d_12 = self.__option_dicts(all_company_df, '申万行业分类一级名称', '申万行业分类二级名称')  # 特定申万一级行业包含的二级行业
        d_23 = self.__option_dicts(all_company_df, '申万行业分类二级名称', '申万行业分类三级名称')  # 特定申万二级行业包含的三级行业
        d_3n = self.__option_dicts(all_company_df, '申万行业分类三级名称', '证券简称')            # 特定申万三级行业包含的上市公司

        d_n1 = self.__value_dicts(all_company_df, '证券简称', '申万行业分类一级名称')             # 特定上市公司的一级行业是什么
        d_n2 = self.__value_dicts(all_company_df, '证券简称', '申万行业分类二级名称')             # 特定上市公司的二级行业是什么
        d_n3 = self.__value_dicts(all_company_df, '证券简称', '申万行业分类三级名称')             # 特定上市公司的三级行业是什么

        d_nc = all_company_df.set_index('证券简称')['证券代码'].to_dict()                # 上市公司名称对应的证券代码

        return all_stock_name, all_industry, d_12, d_23, d_3n, d_n1, d_n2, d_n3, d_nc

    def get_location_and_profit(self, which_industry):

        query_loc = r'SELECT s21.SECNAME, s21.SECCODE, s21.F028V, s21.F036V, s21.F038V, s23.F006N ' \
                    r'FROM stock2100 s21 ' \
                    r'INNER JOIN stock2301 s23 ON s21.SECCODE = s23.SECCODE ' \
                    r'WHERE s21.F034V = "{i}" and s21.CHANGE_CODE <> 2 ' \
                    r'and s23.F001D = "{y}-12-31" and s23.F002V = "071001" ' \
                    r'and s23.CHANGE_CODE <> 2;'.format(i=which_industry, y=newest_fiscal_year())

        return self.__query(query_loc, 'stock2100')

    @staticmethod
    def __option_dicts(df, key_column, value_column):
        raw_dict = df.groupby(key_column)[value_column].unique().to_dict()
        fine_dict = {k: v.tolist() for (k, v) in raw_dict.items()}
        return fine_dict

    @staticmethod
    def __value_dicts(df, key_column, value_column):
        raw_dict = df.groupby(key_column)[value_column].unique().to_dict()
        fine_dict = {k: str(v.squeeze()) for (k, v) in raw_dict.items()}
        return fine_dict

    def get_income(self, secname_list, annual_only=True, reindex_year_and_name=True):

        if annual_only:
            query_income = r'select * from stock2301 where SECNAME in ({l}) and F002V = "071001" and ' \
                           r'CHANGE_CODE <> 2 and F001D REGEXP "12-31";'.format(l=list_to_string(secname_list))
        else:
            query_income = r'select * from stock2301 where SECNAME in ({l}) and F002V = "071001" and ' \
                           r'CHANGE_CODE <> 2;'.format(l=list_to_string(secname_list))

        raw_df = self.__query(query_income, 'stock2301')
        processed = reindex_by_year_and_name(raw_df, secname_list) if reindex_year_and_name else raw_df

        return processed

    def get_kpi(self, secname_list, annual_only=True, reindex_year_and_name=True):

        if annual_only:
            query_kpi = r'select * from stock2303 where SECNAME in ({l}) and F070V = "071001" and CHANGE_CODE <> 2 ' \
                        r'and F001V = "033003" and F069D REGEXP "12-31";'.format(l=list_to_string(secname_list))
        else:
            query_kpi = r'select * from stock2303 where SECNAME in ({l}) and F070V = "071001" and CHANGE_CODE <> 2 ' \
                        r'and F001V = "033003";'.format(l=list_to_string(secname_list))

        raw_df = self.__query(query_kpi, 'stock2303')
        processed = reindex_by_year_and_name(raw_df, secname_list) if reindex_year_and_name else raw_df

        return processed

    def sort_by_revenue(self, secname_list):

        if len(secname_list) == 1:
            return secname_list
        else:
            query_sort = r'select SECNAME from stock2301 where SECNAME in ({l}) ' \
                         r'and F001D = "{y}-12-31" and F002V = "071001" and CHANGE_CODE <> 2 ' \
                         r'order by F006N desc;'.format(l=list_to_string(secname_list), y=newest_fiscal_year())

            return self.__query(query_sort, 'stock2301').squeeze().tolist()

    def latest_price(self, secname_list):

        query_price = r'select SECNAME, TRADEDATE, F002N from stock2401 where SECNAME in ({l});'.format(l=list_to_string(secname_list))
        raw_df = self.__query(query_price, 'stock2401')

        raw_df['昨收盘'] = raw_df['昨收盘'].apply(pd.to_numeric)

        # 取每个公司的最后一行，也就是最新的价格。
        # 注意：①有可能会缺少某些公司的价格; ②latest_price_df的顺序和secname_list是不同的，不过没关系。
        latest_price_df = raw_df.groupby('证券简称', sort=False).last().rename_axis('证券简称')

        return latest_price_df

    def history_price(self, secname_list):

        query_price = r'select SECNAME, TRADEDATE, F002N, F010N, F026N from stock2402 where SECNAME in ({l});'.format(l=list_to_string(secname_list))
        raw_df = self.__query(query_price, 'stock2402')
        return raw_df


def translation_dictionary():
    # 数据库中不同的表格（如stock2300, stock2301），字段名称都需要翻译。这个函数读取字段名称翻译（一个excel文件），变成字典。
    # 字典的key是表格名（如stock2300, stock2301），value是这个表格专属的字典。

    translation_excel_path = 'name_dictionary.xlsx'
    sheet_list = ['stock2100', 'stock2300', 'stock2301', 'stock2302', 'stock2303', 'stock2401', 'stock2402']
    sheet_dfs = pd.read_excel(translation_excel_path, sheet_name=sheet_list)

    sheet_dict = {k: v.set_index('英文名称')['中文名称'].to_dict() for (k, v) in sheet_dfs.items()}
    return sheet_dict


def reindex_by_year_and_name(original_df, secname_list):
    # 这个函数按照报告年度、公司列表，对SQL查询到的df进行reindex。
    # 这样reindex之后得到的df是有固定的框架的，不会因为某年公司多或少导致数据不同，适用于收入、运营效率等图

    # 先要copy一下因为不应该改变输入的df，别的函数会用到
    copied_df = original_df.copy()

    # 往前回溯多久，在这里设置。考虑到这个值不太需要经常更改，所以就放函数里了。
    number_of_year = 15

    # 从今年开始，倒数 number_of_year 年
    newest_year = newest_fiscal_year()
    year_list = range(newest_year, newest_year - number_of_year, -1)

    # 把报告年度从完整的日期改为int64格式的年，为reindex做准备
    copied_df['报告年度'] = copied_df['报告年度'].dt.year

    ordered_index = pd.MultiIndex.from_product([year_list, secname_list], names=['报告年度', '证券简称'])
    reindex_df = copied_df.set_index(['报告年度', '证券简称']).reindex(index=ordered_index)

    return reindex_df


def list_to_string(a_list):
    list_with_quotation = ['\"' + a_item + '\"' for a_item in a_list if a_item]
    return r', '.join(list_with_quotation)


def newest_fiscal_year():

    # 4月以后，去年年报都出来了，因此newest year是去年；反之，newest year是前年
    this_year, this_month = datetime.today().year, datetime.today().month
    newest_year = this_year - 1 if this_month > 4 else this_year - 2

    return newest_year


def add_eps_pe(income_df, price_df):

    # 计算（年度）每股收益的静态、ttm和预测数据。因此需要先获取（季度）“稀释每股收益”的历史数据。
    # “稀释每股收益”那列有时为空，这时就用基本每股收益数据来填充。正好有一个combine_first函数执行这个逻辑
    column_name = 'combined_eps'
    income_df[column_name] = income_df['（二）稀释每股收益'].combine_first(income_df['（一）基本每股收益'])
    income_df.dropna(subset=column_name, inplace=True)

    for current_date, row in income_df.iterrows():

        if current_date.month == 12:
            static = ttm = forcast = row[column_name]
        else:
            try:
                previous_year_end = current_date - pd.offsets.YearEnd(1)
                static = income_df.at[previous_year_end, column_name]

                one_year_ago = datetime(year=current_date.year - 1, month=current_date.month, day=current_date.day)
                ttm = static - income_df.at[one_year_ago, column_name] + income_df.at[current_date, column_name]

                forcast = (income_df.at[current_date, column_name] / income_df.at[one_year_ago, column_name]) * static
            except KeyError:
                static = ttm = forcast = None

        # 四舍五入
        income_df.at[current_date, 'eps_static'] = static
        income_df.at[current_date, 'eps_ttm'] = ttm
        income_df.at[current_date, 'eps_forcast'] = forcast

    # 合并EPS数据和股价数据
    combined_df = pd.merge_ordered(left=price_df,
                                   right=income_df,
                                   left_on=price_df.index,
                                   right_on=income_df.index,
                                   fill_method='ffill').dropna(subset='eps_ttm')

    # 改名字、设置index、删除没有价格的行（往往在上市之前，保留也没意义）
    combined_df = combined_df.rename(columns={'key_0': '交易日期', '证券简称_x': '证券简称'}).set_index('交易日期')
    combined_df = combined_df.dropna(subset='昨日收盘价')

    # 计算三种PE
    combined_df['pe_static'] = combined_df['昨日收盘价'].divide(combined_df.eps_static)
    combined_df['pe_ttm'] = combined_df['昨日收盘价'].divide(combined_df.eps_ttm)
    combined_df['pe_forcast'] = combined_df['昨日收盘价'].divide(combined_df.eps_forcast)

    # 对于计算出来的几个值，设置
    combined_df = combined_df.round({'eps_ttm': 2, 'eps_forcast': 2, 'pe_static': 1, 'pe_ttm': 1, 'pe_forcast': 1})

    return combined_df


def actual_growth(data_series, number_of_year=None):

    data_series.dropna(inplace=True)

    if data_series.empty:
        return '过去{:>2}年：无数据'.format(number_of_year)

    # 如果最后一行正好是年报（12月的），则这行就算最后一年。如果不是，就要往前找一年。比如最后一行是2022年6月，就要找2021年12月31日了
    last_report = data_series.index[-1]
    last_year_end = last_report if last_report.month == 12 else last_report - pd.offsets.YearEnd(1)

    if number_of_year:
        first_year_end = last_year_end - pd.offsets.YearEnd(number_of_year)
    else:
        first_report = data_series.index[0]
        first_year_end = first_report if first_report.month == 12 else first_report + pd.offsets.YearEnd(1)
        number_of_year = last_year_end.year - first_year_end.year

        if number_of_year == 0:
            return '仅有一年数据，无法计算'

    try:
        last_year_data = data_series.loc[last_year_end]
        first_year_data = data_series.loc[first_year_end]
    except KeyError:
        return '过去{:>2}年：无数据'.format(number_of_year)

    else:
        if last_year_data <= 0:
            string_result = '过去{:>2}年：去年亏损'.format(number_of_year)
        elif first_year_data <= 0:
            string_result = '过去{:>2}年：{:>2}年前亏损'.format(number_of_year, number_of_year)
        else:
            compound_growth_rate = (last_year_data / first_year_data) ** (1 / number_of_year) - 1
            string_result = '过去{:>2}年：{:.1%}'.format(number_of_year, compound_growth_rate)

    return string_result


def suggested_growth(pe, discount_rate=0.15):
    # PE = 1 /（折现率 - 增长率）
    # 公式来源：https://mp.weixin.qq.com/s/nb1tFWllUXT-jrCmZRBvNA

    if not pe or isinstance(pe, str):
        string_result = 'PE不存在'
    else:
        if pe <= 0:
            string_result = 'PE为负数'
        else:
            growth_rate = discount_rate - 1/pe
            string_result = '{:.1%}'.format(growth_rate)

    return string_result


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


def income_chart(data_df, title):

    change_names = {'归属于母公司所有者的净利润': '归母净利润', '扣除非经常性损益后的净利润(2007版)': '扣非净利润', '经营活动现金流量净额': '经营现金流'}
    needed_col = ['营业收入', '营业利润', '利润总额', '净利润', '归母净利润', '扣非净利润', '经营现金流']

    df = pick_and_rename(data_df, needed_col, change_names)

    return bar_figure(yi(df), title, y_cols=needed_col, y_title='单位：亿')


def cost_chart(data_df, chart_name):

    change_names = {'一、营业总收入': '营业总收入', '其中：营业成本': '营业成本', '营业税金及附加': '营业税', '加：公允价值变动净收益': '公允价值变动收益'}

    revenue_col = ['营业总收入']
    subtract_col = ['财务费用', '研发费用', '管理费用', '销售费用', '营业税', '利息支出', '营业成本']
    supplement_col = ['投资收益', '公允价值变动收益', '信用减值损失（2019格式）', '资产减值损失（2019格式）', '资产减值损失', '资产处置收益', '其它收入']

    df = pick_and_rename(data_df, revenue_col + subtract_col + supplement_col, change_names)

    # 这里跟一个squeeze()是为了把只有1个column的dataframe变成series，否则后面得到percentage_df那一步会全是Nan。
    revenue = df[revenue_col].squeeze()
    subtract_part = df[subtract_col]

    nominal_operating_profit = revenue - subtract_part.sum(axis=1)
    nominal_operating_profit.name = '名义营业利润'

    # 注意这里对补充部分（supplement_part）统一取负数。因为cost分析这里的各种cost(subtract_col那些列)在报表里都是正数，
    # 而supplement_part这些补充营业利润的项目在报表里也是正数。这里特意变成了负数以把他们这些项目体现在横轴之下，表现为对nominal营业利润的补充。
    # 同时注意，supplement_part要跟dropna，因为有时候有些列是没有数值的，直接取负就会报错。
    supplement_part = - df[supplement_col].dropna(axis='columns', how='all')

    combined_df = pd.concat([nominal_operating_profit, subtract_part, supplement_part], axis=1)
    percentage_df = combined_df.divide(revenue, axis='index').round(2)

    needed_col = percentage_df.columns
    return bar_figure(percentage_df, chart_name, y_cols=needed_col, y_percent=True, stack=True)


def efficiency_chart(data_df, chart_name):
    change_names = {'总资产报酬率': 'ROA', '加权平均净资产收益率': '加权平均ROE'}
    needed_col = ['毛利率', '营业利润率', '净利润率', 'ROA', '加权平均ROE']

    df = pick_and_rename(data_df, needed_col, change_names).round(1)

    return bar_figure(df, chart_name, y_cols=needed_col)


def warren_touch_chart(raw_df, company_name):
    change_names = {'营运资金': '运营资本'}
    needed_col = ['运营资本', '固定资产', '营业收入', '营业利润']

    df = pick_and_rename(raw_df, needed_col, change_names).dropna(axis=0, how='all').dropna(axis=1, how='all')

    # list(df)可以获得df的columns的名字列表。因为上一行有可能会删除一些完全没有数据的列，所以用list(df)获得更新后的列名
    return area_figure(df, company_name, y_cols=list(df))


def annual_report_link(sec_code):
    return 'http://money.finance.sina.com.cn/corp/go.php/vCB_Bulletin/stockid/{c}/page_type/ndbg.phtml'.format(c=sec_code)


def bar_figure(df, title, y_cols, stack=False, y_percent=False, y_title=''):

    df = df.fillna(0).reset_index()
    the_bar_mode = 'relative' if stack else 'group'
    time = '报告年度' if '报告年度' in df else '交易日期'

    if is_single_company(df):
        fig = px.bar(data_frame=df, x=time, y=y_cols, title=title, opacity=0.8)
    else:
        fig = px.bar(data_frame=df, x='证券简称', y=y_cols, animation_frame='报告年度', title=title, opacity=0.8)

    fig.layout.update(
        xaxis_title='',
        yaxis_title=y_title,
        title=dict(x=0.5),
        barmode=the_bar_mode,
    )

    if y_percent:
        fig.layout.update(yaxis=dict(tickformat='.0%'))

    return fig


def area_figure(df, title, y_cols):

    df.reset_index(inplace=True)
    fig = px.area(data_frame=df, x='报告年度', y=y_cols, title=title)
    fig.layout.update(xaxis_title='', yaxis_title='', title=dict(x=0.5))

    return fig


def map_figure(df, title):

    my_token = 'pk.eyJ1IjoibGVpbGl1bGl1bGVpIiwiYSI6ImNsaGczamh5dTBleHgzaXBpM202MXI1ZHAifQ.IP6oZFtsoqtYtAaLnwboWQ'
    px.set_mapbox_access_token(my_token)

    label_dict = {'申万行业分类二级名称': '申万二级行业', '申万行业分类三级名称': '申万三级行业', 'F006N': '总收入'}
    df['城市+公司'] = df['所属城市'].str.cat(df['证券简称'], sep='<br>')

    range_bins = [0, 1.e8, 1.e9, 1.e10, 1.e11, 1.e13]
    label_list = [1, 2, 6, 12, 20]
    df['收入级别'] = pd.cut(df.F006N, bins=range_bins, labels=label_list, include_lowest=True)
    updated_title = '申万一级行业：{}'.format(title)

    fig = px.scatter_mapbox(data_frame=df, title=updated_title, lat='latitude', lon='longitude',
                            text='城市+公司', hover_name='证券简称', labels=label_dict,
                            color='申万行业分类三级名称', animation_frame='申万行业分类二级名称', animation_group='申万行业分类三级名称',
                            size='收入级别', size_max=20, zoom=4, opacity=0.5,
                            width=1500, height=1000, center=dict(lat=35, lon=105))

    fig.layout.update(title=dict(x=0.5))

    return fig


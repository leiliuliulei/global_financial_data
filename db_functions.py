import requests
import pandas as pd
import plotly.express as px
from datetime import datetime


class MySQLQuery(object):

    def __init__(self, engine, dictionary_book):
        self.__engine = engine
        self.__name_book = dictionary_book

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


def static_ttm_forcast(data_series, by_which_date):

    cut_series = data_series[:by_which_date]

    last_report = cut_series.index[-1]
    one_year_ago = last_report - pd.offsets.Day(365)
    previous_year_end = last_report - pd.offsets.YearEnd(1)

    if last_report.month == 12:
        static = ttm = forcast = cut_series.iloc[-1]

    else:
        try:
            static = cut_series.loc[previous_year_end]
            ttm = static - cut_series.loc[one_year_ago] + cut_series.loc[last_report]
            forcast = (cut_series.loc[last_report] / cut_series.loc[one_year_ago]) * static
        except KeyError:
            static = ttm = forcast = None

    # 四舍五入
    static, ttm, forcast = [round(v, 2) if v else None for v in [static, ttm, forcast]]

    return static, ttm, forcast


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


def income_chart(data_df, chart_name):

    new_names = {'归属于母公司所有者的净利润': '归母净利润', '扣除非经常性损益后的净利润(2007版)': '扣非净利润', '经营活动现金流量净额': '经营现金流'}
    needed_col = ['营业收入', '营业利润', '利润总额', '净利润', '归母净利润', '扣非净利润', '经营现金流']

    df = pick_and_rename(data_df, needed_col, new_names)

    return bar_figure(yi(df), '收入对比：{c}'.format(c=chart_name), y_cols=needed_col, y_title='单位：亿')


def cost_chart(data_df, chart_name):

    new_names = {'一、营业总收入': '营业总收入', '其中：营业成本': '营业成本', '营业税金及附加': '营业税', '加：公允价值变动净收益': '公允价值变动收益'}

    revenue_col = ['营业总收入']
    subtract_col = ['财务费用', '研发费用', '管理费用', '销售费用', '营业税', '利息支出', '营业成本']
    supplement_col = ['投资收益', '公允价值变动收益', '信用减值损失（2019格式）', '资产减值损失（2019格式）', '资产减值损失', '资产处置收益', '其它收入']

    df = pick_and_rename(data_df, revenue_col + subtract_col + supplement_col, new_names)

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
    return bar_figure(percentage_df, '成本拆解：{c}'.format(c=chart_name), y_cols=needed_col, y_percent=True, stack=True)


def efficiency_chart(data_df, chart_name):
    new_names = {'总资产报酬率': 'ROA', '加权平均净资产收益率': '加权平均ROE'}
    needed_col = ['毛利率', '营业利润率', '净利润率', 'ROA', '加权平均ROE']

    df = pick_and_rename(data_df, needed_col, new_names).round(1)

    return bar_figure(df, '运营效率对比：{c}'.format(c=chart_name), y_cols=needed_col)


def warren_touch_chart(raw_df, company_name):
    new_names = {'营运资金': '运营资本'}
    needed_col = ['运营资本', '固定资产', '营业收入', '营业利润']

    df = pick_and_rename(raw_df, needed_col, new_names).dropna(axis=0, how='all').dropna(axis=1, how='all')

    # list(df)可以获得df的columns的名字列表。因为上一行有可能会删除一些完全没有数据的列，所以用list(df)获得更新后的列名
    return area_figure(df, '运营资产弹性:{c}'.format(c=company_name), y_cols=list(df))


def balance_sheet_chart(raw_df):

    current_asset_total = ['流动资产合计']
    current_asset_cols = ['货币资金', '交易性金融资产', '衍生金融资产', '一年内到期的非流动资产', '应收票据', '应收账款', '预付款项', '存货']

    fixed_asset_total = ['非流动资产合计']
    fixed_asset_cols = ['债权投资', '长期股权投资', '固定资产', '在建工程', '无形资产', '商誉', '递延所得税资产']

    needed_cols = current_asset_total + current_asset_cols + fixed_asset_total + fixed_asset_cols
    df = pick_and_rename(raw_df, needed_cols)

    df['现金类'] = df['货币资金'] + df['交易性金融资产'] + df['衍生金融资产'] + df['一年内到期的非流动资产']
    df['应收预付'] = df['应收票据'] + df['应收账款'] + df['预付款项']
    df['其他流动资产'] = df['流动资产合计'] - df['现金类'] - df['应收预付'] - df['存货']

    needed_cols = ['现金类', '应收预付', '存货', '其他流动资产']
    combined_df = df[needed_cols]
    # df['其他流动资产'] = df[current_asset_total] - df[current_asset_cols].sum(axis=1)
    # df['其他固定资产'] = df[fixed_asset_total] - df[fixed_asset_cols].sum(axis=1)
    # other_current_asset = df[current_asset_total].squeeze() - df[current_asset_cols].sum(axis=1)
    # other_current_asset.name = '其他流动资产'
    #
    # other_fixed_asset = df[fixed_asset_total].squeeze() - df[fixed_asset_cols].sum(axis=1)
    # other_fixed_asset.name = '其他固定资产'

    # combined_df = pd.concat([df[current_asset_cols], other_current_asset, df[fixed_asset_cols], other_fixed_asset], axis=1)
    # needed_cols = current_asset_cols + ['其他流动资产'] + fixed_asset_cols + ['其他固定资产']

    return bar_figure(yi(combined_df), '资产表', y_cols=needed_cols, stack=True)


def invested_capital(raw_df, company_name):

    new_names = {'营运资金': '运营资本', '四、利润总额': '税前利润', '购建固定资产、无形资产和其他长期资产支付的现金': '固定资产投入'}
    needed_col = ['运营资本', '固定资产投入', '税前利润']
    df = pick_and_rename(raw_df, needed_col, new_names).dropna(how='all')

    start_year = df['报告年度'].min()
    start_workingCapital = df['运营资本']

    end_year = raw_df['报告年度'].max()


def annual_report_link(sec_code):
    return 'http://money.finance.sina.com.cn/corp/go.php/vCB_Bulletin/stockid/{c}/page_type/ndbg.phtml'.format(c=sec_code)


def pe_calc(company_name, engine):

    oldest_day = '2012-12-31'

    # s1.F032N：稀释每股收益；s3.F006N: 扣除非经常性损益每股收益
    # 如果没有 F070V = '071001' 这个过滤条件，会得到一些重复的数据。有的是当年提供的，有的是重述往年的。
    sql_string_for_eps = r"select s1.ENDDATE, s1.DECLAREDATE, s1.F032N, s3.F006N " \
                         r"from stock2301 s1 straight_join stock2303 s3 " \
                         r"on(s1.SECNAME = s3.SECNAME and s1.ENDDATE = s3.ENDDATE) " \
                         r"where s1.SECNAME = '{n}' and s1.CHANGE_CODE <> 2 and s3.CHANGE_CODE <> 2 " \
                         r"and s1.F002V = '071001' and s3.F070V = '071001'" \
                         r"order by S1.ENDDATE".format(n=company_name)

    df_eps = mysql_query(sql_string_for_eps, engine)
    df_eps.ENDDATE = pd.to_datetime(df_eps.ENDDATE)

    # 这是包含所有季度数据、年度数据的原始 time series
    raw_ts = df_eps.set_index('ENDDATE').squeeze().truncate(before=oldest_day)

    # 提取出年度数据（xxxx-12-31那几行）
    annual = raw_ts.reindex(pd.date_range(start=oldest_day, end=datetime.today(), freq='Y'))

    # 上面的 annual 还查最后一行，那就是从最近一个12月31号直到今天的。所以在这个series最后补充一行数据：up_to_today
    index_of_today = pd.Series(data=None, dtype=pd.Float64Dtype, index=[datetime.today()])
    annual = pd.concat([annual, index_of_today])
    quarter = pd.concat([raw_ts, index_of_today])

    # 按照 business day ('B') 进行上采样并填上(ffill)最相邻的EPS数据
    business_day_annual_eps = annual.resample('B').ffill()
    business_day_quarter_eps = quarter.resample('B').ffill()

    # 获取历史价格。F002N：收盘价
    sql_string_for_price = r"select TRADEDATE, F002N from stock2402 where SECNAME = '{n}';".format(n=company_name)

    df_price = pd.read_sql_query(sql_string_for_price, engine)
    df_price.TRADEDATE = pd.to_datetime(df_price.TRADEDATE)

    business_day_price = df_price.set_index('TRADEDATE').squeeze().truncate(before=oldest_day)

    df_dictionary = {'静态PE': business_day_annual_eps, '动态PE': business_day_quarter_eps, '价格': business_day_price}
    df = pd.concat(df_dictionary, axis=1).dropna(axis='index', how='any')


def bar_figure(df, title, y_cols, stack=False, y_percent=False, y_title=''):

    df = df.fillna(0).reset_index()
    the_bar_mode = 'relative' if stack else 'group'

    if is_single_company(df):
        fig = px.bar(data_frame=df, x='报告年度', y=y_cols, title=title, opacity=0.8)
    else:
        fig = px.bar(data_frame=df, x='证券简称', y=y_cols, animation_frame='报告年度', title=title, opacity=0.8)

    fig.layout.update(xaxis_title='', yaxis_title=y_title, title=dict(x=0.5), barmode=the_bar_mode)
    # fig.layout.update(xaxis_autorange='reversed')

    if y_percent:
        fig.layout.update(yaxis=dict(tickformat='.0%'))

    return fig


def area_figure(df, title, y_cols):

    df.reset_index(inplace=True)
    fig = px.area(data_frame=df, x='报告年度', y=y_cols, title=title)
    fig.layout.update(xaxis_title='', yaxis_title='', title=dict(x=0.5))

    return fig


class CnInfoAPI:

    def __init__(self, key, secret, dictionary_book, name_to_code_dict):
        self.__token = self.__get_token(key, secret)
        self.__name_book = dictionary_book
        self.__d_nc = name_to_code_dict

    def get_price(self, secname_list):

        result_list = [self.__current_price_base(self.__d_nc[name]) for name in secname_list]
        combined = pd.concat(result_list)
        return combined

    def __current_price_base(self, code):

        price_url = 'http://webapi.cninfo.com.cn/api/stock/p_stock2402'

        last_trading_day = (datetime.today() - pd.offsets.BDay()).strftime('%Y-%m-%d')
        post_data = {'scode': code, 'sdate': last_trading_day, 'edate': last_trading_day, '@column': 'SECNAME,TRADEDATE,F002N'}

        downloaded = self.__cninfo_api(price_url, post_data)
        renamed = downloaded.rename(columns=self.__name_book['stock2402']) if not downloaded.empty else downloaded
        return renamed

    def __get_token(self, key, secret):

        url = 'http://webapi.cninfo.com.cn/api-cloud-platform/oauth2/token'
        post_data = {'grant_type': 'client_credentials', 'client_id': key, 'client_secret': secret}

        the_token = self.__cninfo_api(url, post_data, result_keyword='access_token', return_as_dataframe=False)

        if the_token:
            print('token获取成功')
        else:
            print('token获取失败')

        return the_token

    def __cninfo_api(self, url, post_data, result_keyword='records', return_as_dataframe=True):

        # 对一般的查询，肯定需要附加token。但唯独对获取token的查询不需要（因为此时还没有token）
        try:
            post_data.update(access_token=self.__token)
        except AttributeError:
            pass

        raw = requests.post(url=url, data=post_data).json()

        try:
            result = raw[result_keyword]
        except KeyError:
            print('数据获取错误，获取到的原始数据：', raw)
            result = None

        result = pd.DataFrame(result) if return_as_dataframe else result

        return result


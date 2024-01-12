import toml
import pandas as pd
import plotly.express as px
from datetime import datetime
import plotly.graph_objs as go
from database import FinancialDatabase
from plotly.subplots import make_subplots


class DataA(object):

    def __init__(self, secname):

        self.db = FinancialDatabase()
        self._secname = secname
        self._single = True if isinstance(secname, str) else False

        self._df = None

        self._yi = True
        self._year_only = True

        self._year_col = '截止日期'
        self._name_col = '证券简称'
        self._y_col = []

        self._title = ''
        self._x_title = ''
        self._y_title = ''
        self._barmode = 'group'

        self.chart_config = toml.load('configuration.toml')['A']

        # self._efficiency = query_setting('stock2303', ['证券简称', '截止日期', '毛利率', '营业利润率', '净利润率', 'ROA', '加权平均ROE'])
        # self._warren = query_setting('stock2303', ['证券简称', '截止日期', '运营资本', '固定资产', '营业收入', '营业利润'])

    def segment_info(self):
        conf = self.chart_config['segment']
        df = self.db.query_without_sort(table=conf['table'], column_list=conf['db_col'], secname=None)

        stocks = df['证券简称'].unique().tolist()
        industry_1 = df['一级行业名称'].unique().tolist()

        # d代表dict，1、2、3代表一、二、三级行业，n代表name（上市公司名称）, c代表code（上市公司代码）
        if '三级行业名称' in df:
            d_12 = include_dict(df, '一级行业名称', '二级行业名称')     # 特定 申万一级行业 包含的 二级行业
            d_23 = include_dict(df, '二级行业名称', '三级行业名称')     # 特定 申万二级行业 包含的 三级行业
            d_3n = include_dict(df, '三级行业名称', '证券简称')        # 特定 申万三级行业 包含的 上市公司

            df.set_index('证券简称', inplace=True)
            d_n1 = df['一级行业名称'].to_dict()                       # 特定上市公司的一级行业是什么
            d_n2 = df['二级行业名称'].to_dict()                       # 特定上市公司的二级行业是什么
            d_n3 = df['三级行业名称'].to_dict()                       # 特定上市公司的三级行业是什么
        else:
            d_12 = include_dict(df, '一级行业名称', '二级行业名称')     # 特定 一级行业 包含的 二级行业
            d_23 = None                                             # 港股没有三级行业
            d_3n = include_dict(df, '二级行业名称', '证券简称')        # 特定 二级行业 包含的 上市公司。为了拉齐A股的变量名仍叫d_3n

            df.set_index('证券简称', inplace=True)
            d_n1 = df['一级行业名称'].to_dict()                       # 特定上市公司的一级行业是什么
            d_n2 = df['二级行业名称'].to_dict()                       # 特定上市公司的二级行业是什么
            d_n3 = None                                             # 港股没有三级行业

        d_nc = df['证券代码'].to_dict()                              # 上市公司名称对应的证券代码

        segment_info_dict = {
            'stocks': stocks,
            'industry_1': industry_1,
            'd_12': d_12,
            'd_23': d_23,
            'd_3n': d_3n,
            'd_n1': d_n1,
            'd_n2': d_n2,
            'd_n3': d_n3,
            'd_nc': d_nc}

        return segment_info_dict

    def sort(self):
        if len(self._secname) == 1:
            return self._secname
        else:
            conf = self.chart_config['sort']
            return self.db.sort_income(table=conf['table'], column_list=conf['db_col'], secname=self._secname).squeeze().tolist()

    def income_fig(self, title):

        conf = self.chart_config['income']
        self._df = self._basic_data(conf)

        self._y_col = conf['y_col']
        self._y_title = '单位：亿'
        self._title = title

        return self._bar()

    def cost_fig(self, title):

        conf = self.chart_config['cost']

        title_col = conf['title_col']
        revenue_col = conf['revenue_col']
        subtract_col = conf['subtract_col']
        supplement_col = conf['supplement_col']

        df = self._basic_data(conf).set_index(title_col)

        # 这里跟一个squeeze()是为了把只有1个column的dataframe变成series，否则后面得到percentage_df那一步会全是Nan。
        revenue = df[revenue_col].squeeze()
        subtract_part = df[subtract_col].dropna(axis='columns', how='all')

        nominal_operating_profit = revenue - subtract_part.sum(axis=1)
        nominal_operating_profit.name = '名义营业利润'

        # 注意这里对补充部分（supplement_part）统一取负数。因为cost分析这里的各种cost(subtract_col那些列)在报表里都是正数，
        # 而supplement_part这些补充营业利润的项目在报表里也是正数。这里特意变成了负数以把他们这些项目体现在横轴之下，表现为对nominal营业利润的补充。
        # 同时注意，supplement_part要跟dropna，因为有时候有些列是没有数值的，直接取负就会报错。
        supplement_part = -df[supplement_col].dropna(axis='columns', how='all')

        combined_df = pd.concat([nominal_operating_profit, subtract_part, supplement_part], axis=1)
        percentage_df = combined_df.divide(revenue, axis='index').round(2)

        self._title = title
        self._y_col = percentage_df.columns
        self._barmode = 'relative'
        self._yi = False
        self._df = percentage_df.reset_index()

        fig = self._bar()
        fig.layout.update(yaxis=dict(tickformat='.0%'))

        return fig

    def efficiency_fig(self, title):

        conf = self.chart_config['efficiency']
        self._df = self._basic_data(conf)

        self._yi = False

        self._y_col = conf['y_col']
        self._title = title

        return self._bar()

    def warren_fig(self):

        conf = self.chart_config['warren']
        self._df = self._basic_data(conf).set_index(self._year_col).dropna(axis=0, how='all').dropna(axis=1, how='all')

        self._y_col = self._df.columns
        self._df.reset_index(inplace=True)

        return self._area()

    def valuation_fig(self):

        # eps数据按季度更新，价格和pe按日更新
        pe_df = self._history_pe()

        start_date, end_date = pe_df.index[0], pe_df.index[-1]
        total_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)

        # 因为原始的 valuation_df 的颗粒度是季度，但是画图的时候发现，按季度画图不好看，抖动太多。于是这里把它按照年度重采样。
        # A-May 表示年度采样而且以May做为每年的开头。这样的好处在于五月份刚刚把所有的年报更新完，五月的价格考虑了最新业绩。
        if total_months > 120:
            chart_df = pe_df.resample('A-MAY', convention='end').nearest().loc[: datetime.today()]
        elif 24 < total_months < 120:
            chart_df = pe_df.resample('BQ', convention='end').nearest().loc[: datetime.today()]
        else:
            chart_df = pe_df.resample('BM', convention='end').nearest().loc[: datetime.today()]

        # 画图。尝试过画在一起但大小差异很大，不好看。因此分成三个独立的图。
        fig = make_subplots(rows=3, cols=1, subplot_titles=['每股收益（TTM）', '股价', '市盈率（TTM）'], vertical_spacing=0.1)

        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['eps_ttm'], mode='lines+markers+text', text=chart_df['eps_ttm'], textposition='top center'), row=1, col=1)
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['收盘价'], mode='lines+markers+text', text=chart_df['收盘价'].round(1), textposition='top center'), row=2, col=1)
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['pe_ttm'], mode='lines+markers+text', text=chart_df['pe_ttm'].round(), textposition='top center'), row=3, col=1)

        max_width = 1400
        look_good_width = len(chart_df) * 60

        fig.update_layout(height=800, width=min(look_good_width, max_width), title_text=self._secname, showlegend=False)

        return fig

    def growth_table(self):

        # 计算精细颗粒度的EPS和PE
        eps_df_dict = self._get_eps(refine_eps=False)
        pe_df_dict = self._history_pe()

        # 获得证券代码字典
        conf = self.chart_config['code']
        code_dict = self.db.query_without_sort(conf['table'], conf['db_col'], self._secname).set_index('证券简称')['证券代码'].to_dict()

        # 生成数据
        table_data = [self._one_data_row(name, code_dict[name], eps_df_dict[name], pe_df_dict[name]) for name in self._secname]
        tooltip_data = [self._one_tooltip_row(pe_df_dict[name]) for name in self._secname]

        return table_data, tooltip_data

    def industry_map(self, industry_2_name):

        conf = self.chart_config['geo_setting']
        px.set_mapbox_access_token(conf['mapbox_token'])

        # 获取数据
        geo_df = pd.read_excel(conf['geo_file_path'])
        location_df = self.db.query_location(industry_2_name)

        # 附加GPS信息
        merged_df = location_df.merge(geo_df, left_on='所属城市', right_on='区域名称').dropna().rename(columns={'F006N': '总收入'})

        # 整理数据
        merged_df['城市+公司'] = merged_df['所属城市'].str.cat(merged_df['证券简称'], sep='<br>')

        # 根据总收入，划分几个等级，体现为图里的size参数，即bubble大小
        range_bins = [0, 1.0e8, 1.0e9, 1.0e10, 1.0e11, 1.0e13]
        label_list = [1, 2, 6, 12, 20]
        merged_df['收入级别'] = pd.cut(merged_df['总收入'], bins=range_bins, labels=label_list, include_lowest=True)

        # 画图
        updated_title = f'申万二级行业：{industry_2_name}'
        fig = px.scatter_mapbox(data_frame=merged_df, title=updated_title, lat='latitude', lon='longitude',
                                text='城市+公司', hover_name='证券简称', color='三级行业名称', size='收入级别', size_max=20,
                                zoom=4, opacity=0.5, width=1400, height=1000, center=dict(lat=35, lon=105))

        fig.layout.update(title=dict(x=0.5))

        return fig

    @staticmethod
    def _one_data_row(secname, seccode, eps_df, pe_df):

        company_link = markdown_cell(text=secname, link=sina_report_link(seccode))

        if isinstance(pe_df, str):
            return {'公司': company_link, '价格': pe_df, 'PE': pe_df, 'PE暗示的成长性': pe_df, '实际EPS成长性': pe_df}

        if pe_df is None or pe_df.empty:
            return {'公司': company_link, '价格': None, 'PE': None, 'PE暗示的成长性': None, '实际EPS成长性': None}

        pe_last_row = pe_df.iloc[-1].squeeze()
        last_price = pe_last_row['收盘价']
        eps_series = eps_df.set_index('截止日期')['稀释每股收益'].squeeze()

        # 当前PE暗示的成长性
        growth_s = suggested_growth(pe_last_row.pe_static)
        growth_t = suggested_growth(pe_last_row.pe_ttm)
        growth_f = suggested_growth(pe_last_row.pe_forcast)

        # 实际的成长性
        if eps_df is None or eps_df.empty:
            actual_inf = actual_10 = actual_5 = '未能获取EPS数据'
        else:
            actual_inf = actual_growth(eps_series)
            actual_10 = actual_growth(eps_series, 10)
            actual_5 = actual_growth(eps_series, 5)

        data_row = {
            '公司': company_link,
            '价格': last_price,
            'PE': f'静态：{pe_last_row.pe_static}\nTTM：{pe_last_row.pe_ttm}\n预测：{pe_last_row.pe_forcast}',
            'PE暗示的成长性': f'静态：{growth_s}\nTTM：{growth_t}\n预测：{growth_f}',
            '实际EPS成长性': f'{actual_inf}\n{actual_10}\n{actual_5}'
        }

        return data_row

    @staticmethod
    def _one_tooltip_row(pe_df):

        if isinstance(pe_df, str) or pe_df is None or pe_df.empty:
            return {'公司': None, '价格': None, 'PE': None, 'PE暗示的成长性': None, '实际EPS成长性': None}

        pe_last_row = pe_df.iloc[-1].squeeze()
        pe_time = pe_df.index[-1]

        tooltip_row = {
            'PE': {'value': f'静态 EPS：{pe_last_row.eps_static}\\\nTTM EPS：{pe_last_row.eps_ttm}\\\n预测 EPS：{pe_last_row.eps_forcast}',
                   'type': 'markdown'},
            '价格': f'日期：{pe_time.strftime("%Y-%m-%d")}'
            }

        return tooltip_row

    def _basic_data(self, configuration, annual=True):
        return self.db.query_secname(table=configuration['table'], column_list=configuration['db_col'],
                                     secname=self._secname, annual=annual)

    def _bar(self):

        if self._year_only:
            self._df[self._year_col] = self._df[self._year_col].dt.year

        if self._yi:
            self._df[self._y_col] = self._df[self._y_col].divide(1.0e8).round(1)

        if self._single:
            fig = px.bar(data_frame=self._df, x=self._year_col, y=self._y_col, title=self._secname, opacity=0.8)
        else:
            fig = px.bar(data_frame=self._df, x=self._name_col, y=self._y_col, animation_frame=self._year_col, title=self._title, opacity=0.8)

        fig.layout.update(xaxis_title=self._x_title, yaxis_title=self._y_title, barmode=self._barmode, title=dict(x=0.5))

        return fig

    def _area(self):

        if self._year_only:
            self._df[self._year_col] = self._df[self._year_col].dt.year

        fig = px.area(data_frame=self._df, x=self._year_col, y=self._y_col, title=self._secname)
        fig.layout.update(xaxis_title=self._x_title, yaxis_title=self._y_title, title=dict(x=0.5))

        return fig

    def _history_pe(self):

        # 获取精细化EPS以及每日价格数据
        eps_df = self._get_eps(refine_eps=True)
        conf = self.chart_config['price']
        price_df = self.db.query_without_sort(table=conf['table'], column_list=conf['db_col'], secname=self._secname)

        if self._single:
            return calc_pe(eps_df, price_df)
        else:
            eps_df_dict = eps_df
            price_df_dict = {name: price_df.query('证券简称==@name') for name in self._secname}
            pe_dict = {name: calc_pe(eps_df_dict[name], price_df_dict[name]) for name in self._secname}
            return pe_dict

    def _get_eps(self, refine_eps=True):

        def refine(nominal_eps_df):
            # 这个函数只接受单一公司的eps_df
            if nominal_eps_df is None or nominal_eps_df.empty:
                return None
            else:
                # 推算出三个精细EPS数据（静态、滚动、预测），按照 '公告日期' 重建表格，然后去重（因为有时候年报和一季报同一天发布，此时仅保留一季报那行）
                nominal_eps_df = nominal_eps_df.set_index('截止日期').dropna()
                refined_eps_data = [self._eps_at_date(nominal_eps_df, a_date) for a_date in nominal_eps_df.index]
                refined_eps_df = pd.DataFrame.from_records(data=refined_eps_data, index=nominal_eps_df['公告日期'])
                refined_eps_df = refined_eps_df.reset_index().drop_duplicates(subset='公告日期', keep='first').dropna()
                return refined_eps_df

        # 从数据库获取EPS数据
        eps_df = self._basic_data(self.chart_config['EPS'], annual=False)

        if self._single:
            if refine_eps:
                return refine(eps_df)
            else:
                return eps_df

        # 若传入了secname_list，则输出一个字典。遍历self._secname保证了字典的完备性，如果用eps_df.groupby('证券简称')有可能丢失某些公司
        else:
            complete_dict = {name: eps_df.query('证券简称==@name') for name in self._secname}
            if refine_eps:
                return {name: refine(df) for name, df in complete_dict.items()}
            else:
                return complete_dict

    @staticmethod
    def _eps_at_date(eps_df, a_date):

        eps_col = '稀释每股收益'
        current_eps = eps_df.loc[a_date, eps_col]

        if a_date.month == 12:
            static = ttm = forcast = current_eps
        else:
            try:
                previous_year_end = a_date - pd.offsets.YearEnd(1)
                static = eps_df.loc[previous_year_end, eps_col]

                one_year_ago_eps = eps_df.loc[datetime(a_date.year - 1, a_date.month, a_date.day), eps_col]
                ttm = static - one_year_ago_eps + current_eps

                forcast = None if one_year_ago_eps == 0.0 else (current_eps / one_year_ago_eps) * static

            except KeyError:
                static = ttm = forcast = None

        return {'eps_static': static, 'eps_ttm': ttm, 'eps_forcast': forcast}


class DataH(DataA):

    def __init__(self, secname):
        super().__init__(secname)
        self.chart_config = toml.load('configuration.toml')['H']

    def warren_fig(self):

        # 港股的所需数据来自两个表，和A股不一样。因此要重写这个函数。
        conf = self.chart_config['warren']
        df1 = self.db.query_secname(table=conf['table_1'], column_list=conf['db_col_1'], secname=self._secname)
        df2 = self.db.query_secname(table=conf['table_2'], column_list=conf['db_col_2'], secname=self._secname)
        merged_df = df1.merge(df2)

        self._df = merged_df.set_index(self._year_col).dropna(axis=0, how='all').dropna(axis=1, how='all')

        self._y_col = self._df.columns
        self._df.reset_index(inplace=True)

        return self._area()


def include_dict(df, key_column, value_column):
    return df.groupby(key_column)[value_column].apply(set).apply(list).to_dict()


def calc_pe(single_eps_df, single_price_df):

    if single_eps_df is None or single_eps_df.empty:
        return '未能获取EPS历史数据'

    if single_price_df is None or single_price_df.empty:
        return '未能获取价格数据'

    # 合并EPS数据和股价数据。合并后需要再次去重，因为有时公告日期在周末，和交易日期合并时，会多出一行最靠近公告日期的工作日记录，造成重复。
    combined_df = pd.merge_ordered(left=single_eps_df,
                                   right=single_price_df,
                                   left_on='公告日期',
                                   right_on='交易日期',
                                   fill_method='ffill').dropna(subset=['eps_ttm', '收盘价'])

    combined_df.drop_duplicates(subset='交易日期', keep='first', inplace=True)
    combined_df = combined_df.set_index('交易日期').drop(columns='公告日期')

    # 计算三种PE。直接用div不行，因为不支持一列数据（收盘价）分别去除3列。但 rdiv 支持3列数据分别被1列数据反除。注意设置axis。
    eps_cols = ['eps_static', 'eps_ttm', 'eps_forcast']
    pe_cols = ['pe_static', 'pe_ttm', 'pe_forcast']
    combined_df[pe_cols] = combined_df[eps_cols].rdiv(combined_df['收盘价'], axis=0)

    # 四舍五入并返回
    return combined_df.round({'eps_ttm': 2, 'eps_forcast': 2, 'pe_static': 1, 'pe_ttm': 1, 'pe_forcast': 1})


def actual_growth(data_series, number_of_year=None):

    data_series = data_series.dropna()

    if data_series.empty:
        return '过去{:>2}年：无数据'.format(number_of_year)

    first_year, last_year = get_first_last_year(data_series)
    if number_of_year:
        first_year = last_year - pd.offsets.YearEnd(number_of_year)
    else:
        number_of_year = last_year.year - first_year.year
        if number_of_year == 0:
            return '仅有一年数据，无法计算'

    try:
        last_year_data = data_series.loc[last_year]
        first_year_data = data_series.loc[first_year]
    except KeyError:
        return '过去{:>2}年：无数据'.format(number_of_year)

    if last_year_data <= 0:
        string_result = '过去{:>2}年：去年亏损'.format(number_of_year)
    elif first_year_data <= 0:
        string_result = '过去{:>2}年：{:>2}年前亏损'.format(number_of_year, number_of_year)
    else:
        compound_growth_rate = (last_year_data / first_year_data) ** (1 / number_of_year) - 1
        string_result = '过去{:>2}年：{:.1%}'.format(number_of_year, compound_growth_rate)

    return string_result


def suggested_growth(pe, discount_rate=0.15):
    # PE = 1 /（折现率 - 增长率）。 公式来源：https://mp.weixin.qq.com/s/nb1tFWllUXT-jrCmZRBvNA

    if not isinstance(pe, float):
        # if pe is None or isinstance(pe, str):
        string_result = 'PE不存在'
    else:
        if pe <= 0:
            string_result = 'PE为负数'
        else:
            growth_rate = discount_rate - 1 / pe
            string_result = '{:.1%}'.format(growth_rate)

    return string_result


def get_first_last_year(data_series):

    latest_date, earliest_date = data_series.index[0], data_series.index[-1]

    last_year = latest_date if latest_date.month == 12 else latest_date - pd.offsets.YearEnd(1)
    first_year = earliest_date if earliest_date.month == 12 else earliest_date + pd.offsets.YearEnd(1)

    return first_year, last_year


def markdown_cell(text, link):
    return "[{t}]({l})".format(t=text, l=link)


def sina_report_link(sec_code):
    return f'http://money.finance.sina.com.cn/corp/go.php/vCB_Bulletin/stockid/{sec_code}/page_type/ndbg.phtml'

import numpy as np
import toml
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine
from functools import partial
from pypinyin import slug
from collections import ChainMap


class SingleProcessor(object):

    def __init__(self, seccode, raw_eps_df=None, raw_price_df=None, raw_kpi_df=None):

        self.code = seccode

        self._raw_eps = raw_eps_df.query('证券代码==@seccode') if raw_eps_df is not None else None
        self._raw_price = raw_price_df.query('证券代码==@seccode') if raw_price_df is not None else None
        self._raw_kpi = raw_kpi_df.query('证券代码==@seccode') if raw_kpi_df is not None else None

    def pe(self):

        # 如果港股公司还没上市，会体现在没有price数据。这时直接返回空白df。
        if self._raw_eps.empty or self._raw_price.empty:
            empty_info = {'证券代码': [self.code], 'PE': ['nan nan nan']}
            return pd.DataFrame(empty_info)

        # 合并EPS数据和股价数据
        combined_df = pd.merge_ordered(left=self._rolling_eps(),
                                       right=self._raw_price,
                                       left_on='公告日期',
                                       right_on='交易日期',
                                       fill_method='ffill',
                                       suffixes=(None, '_price')).dropna(subset=['eps_ttm', '收盘价'])

        # 合并后需要再次去重，因为有时公告日期在周末，和交易日期合并时，会多出一行最靠近公告日期的工作日记录，造成重复。
        combined_df = combined_df.drop(columns='证券代码_price').drop_duplicates(subset='交易日期', keep='first')

        # 计算三种PE
        price = combined_df['收盘价']
        pe_df = combined_df.assign(pe_static=price / combined_df.eps_static,
                                   pe_ttm=price / combined_df.eps_ttm,
                                   pe_forcast=price / combined_df.eps_forcast)

        pe_df['PE'] = pe_df[['pe_static', 'pe_ttm', 'pe_forcast']].map(to_str).agg(' '.join, axis=1)
        pe_df.drop(columns=['基本每股收益', '稀释每股收益'], inplace=True)

        return pe_df

    def summary_line(self):

        growth_years = [3, 5, 10]
        average_years = [1, 5, 10]

        # 获得这个公司的PE、成长性数据、KPI，合并在一起
        pe_df = self.pe().set_index('证券代码')
        kpi_df = self.average_kpi(average_years)
        growth_df = self.growth_data(growth_years)

        summary_df = pd.concat([pe_df, kpi_df, growth_df], axis=1)

        # 获得上市年限，放到summary_df里
        eps_series = self._raw_eps.set_index('截止日期')['稀释每股收益']
        if eps_series.empty:
            summary_df['上市年限'] = '不到1年'
        else:
            first, last = first_last_year(eps_series)
            summary_df['上市年限'] = last.year - first.year + 1

        # 计算EPS成长性应得的PE估值和当前PE估值的差异。估值gap越大越好（现价买入，未来潜在收益越高）
        average_pe = summary_df['PE'].str.split(expand=True).astype(float).squeeze().mean()
        average_growth = summary_df['EPS成长性'].str.split(expand=True).astype(float).squeeze().mean()

        summary_df[['估值gap', '总收益']] = self._pe_gap(average_growth, average_pe)

        return summary_df

    def growth_data(self, growth_years):
        # 计算过去多年的EPS成长性

        if self._raw_eps.empty:
            empty_info = {'证券代码': [self.code], 'EPS成长性': ['nan nan nan']}
            return pd.DataFrame(empty_info).set_index('证券代码')

        raw_eps = self._raw_eps.set_index('截止日期')

        growth_funcs = [partial(growth_rate, past_n_year=n) for n in growth_years]
        growth_df = self._multi_func(raw_eps, '稀释每股收益', growth_funcs).rename(columns={'稀释每股收益': 'EPS成长性'})

        return growth_df

    def average_kpi(self, average_years):
        # 计算多个指标的多年平均值

        name_dict = {'加权平均ROE': 'ROE', '平均ROE': 'ROE'}      # 由于A股和港股的名称不同，先把名称统一
        kpi_cols = ['毛利率', '净利率', 'ROE', '资产负债比']      # 定义具体求哪些指标的均值

        # 如果公司刚上市不久，会没有kpi数据（目前KPI数据来自定期报告，不来自招股书/预披露报告)，这时返回空白df。格式保持和有数据时一致。
        if self._raw_kpi.empty:
            kpi_nan_data = {col: [''] for col in kpi_cols}
            kpi_nan_data.update({'证券代码': [self.code]})
            kpi_df = pd.DataFrame(kpi_nan_data).set_index('证券代码')

        else:
            raw_kpi_df = self._raw_kpi.set_index('截止日期').rename(columns=name_dict)

            mean_funcs = [partial(n_year_mean, past_n_year=n) for n in average_years]
            kpi_df_list = [self._multi_func(raw_kpi_df, col, mean_funcs) for col in kpi_cols]

            kpi_df = pd.concat(kpi_df_list, axis=1)

        return kpi_df

    def _rolling_eps(self):

        if self._raw_eps.empty:
            return None

        # 设置index，为后续处理做准备
        eps_df = self._raw_eps.drop_duplicates(subset='截止日期').set_index('截止日期').dropna()

        # 推算出三个年度级EPS数据，并生成df
        year_eps_data = [eps_at_date(eps_df['稀释每股收益'], a_date) for a_date in eps_df.index]
        year_eps_df = pd.DataFrame.from_records(data=year_eps_data, index=eps_df.index)

        # 把两个df的数据合并，然后去重（因为有时候年报和一季报同一天发布，此时仅保留一季报那行）
        merged_df = eps_df.merge(year_eps_df, how='left', left_index=True, right_index=True).reset_index().dropna()

        return merged_df

    @staticmethod
    def _multi_func(raw_df, col_name, funcs):

        result_df = raw_df.groupby('证券代码')[col_name].agg(funcs).map(to_str)
        result_df[col_name] = result_df.agg(' '.join, axis=1)

        return result_df[[col_name]]

    @staticmethod
    def _pe_gap(eps_growth_rate, current_pe, wait_year=8):

        # 默认等待8年估值回归，是根据经验，一轮牛熊市的轮回周期。

        if np.isnan(eps_growth_rate) or np.isnan(current_pe) or eps_growth_rate is None or current_pe is None or current_pe <= 0:
            return 'N/A', 'N/A'

        eps_growth_rate = eps_growth_rate / 100
        discount_rate = 0.15

        # 如果增长速度超过折现率（15%），理论估值应该无限大。但为了方便计算，这里给了一个较高的PE，以便函数仍然出一个结果。
        actual_rate = discount_rate - eps_growth_rate

        if actual_rate > 0:
            deserved_pe = 1 / actual_rate
        elif -0.1 < actual_rate <= 0:
            deserved_pe = 50
        elif -0.25 < actual_rate <= -0.1:
            deserved_pe = 75
        else:
            deserved_pe = 100

        the_gap = (deserved_pe / current_pe) ** (1 / wait_year) - 1
        the_gap = the_gap * 100

        eps_growth_rate = eps_growth_rate * 100
        total_gain = the_gap + eps_growth_rate

        the_gap, total_gain = to_str(the_gap, 1), to_str(total_gain, 1)

        return the_gap, total_gain


class Data(object):

    def __init__(self, seccodes, market):

        self._db = FinancialDatabase()
        self.seccodes = seccodes
        self._market = market

        self._year_col = '截止日期'

        dict_a = {
            'income': {'table': 'stock2303',
                       'cols': ['证券简称', '截止日期', '营业收入', '营业利润', '利润总额', '净利润', '归母净利润',
                                '扣非净利润', '经营现金流']},

            'cost': {'table': 'stock2301',
                     'cols': ['证券简称', '截止日期', '营业总收入', '财务费用', '研发费用', '管理费用', '销售费用',
                              '营业税', '利息支出', '营业成本', '投资收益', '公允价值变动收益', '信用减值损失（2019格式）',
                              '资产减值损失（2019格式）', '资产减值损失', '资产处置收益', '其它收入']},

            'efficiency': {'table': 'stock2303',
                           'cols': ['证券简称', '截止日期', '毛利率', '营业利润率', '净利率', 'ROA', '加权平均ROE']},

            'warren': {'table': 'stock2303',
                       'cols': ['截止日期', '运营资本', '固定资产', '营业收入', '营业利润']},

            'eps': {'table': 'stock2301',
                    'cols': ['证券简称', '证券代码', '截止日期', '公告日期', '基本每股收益', '稀释每股收益']},

            'kpi': {'table': 'stock2303',
                    'cols': ['证券代码', '截止日期', '毛利率', '净利率', '加权平均ROE', '资产负债比']},

            'bubble': {'table': 'stock2303',
                       'cols': ['证券简称', '营业收入', '加权平均ROE']},

            'industry': {'table': 'stock2100',
                         'cols': ['证券简称', '证券代码', '一级行业名称', '二级行业名称', '三级行业名称']}

        }

        dict_h = {
            'income': {'table': 'hk4025',
                       'cols': ['证券简称', '截止日期', '报表类别', '营业额', '经营溢利', '除税前经营溢利', '除税后经营溢利',
                                '股东应占溢利', '经营现金流']},

            'cost': {'table': 'hk4024',
                     'cols': ['证券简称', '截止日期', '报表类别', '营业额', '财务成本', '行政费用', '销售及分销成本', '折旧',
                              '贷款利息', '经营开支总额', '特殊项目', '联营公司']},

            'efficiency': {'table': 'hk4025',
                           'cols': ['证券简称', '截止日期', '报表类别', '毛利率', '税前利润率', '净利率', '平均ROA', '平均ROE']},

            'warren': {'table_1': 'hk4023', 'cols_1': ['截止日期', '报表类别', '运营资本', '固定资产'],
                       'table_2': 'hk4024', 'cols_2': ['截止日期', '报表类别', '营业额', '除税前经营溢利']},

            'eps': {'table': 'hk4025',
                    'cols': ['证券简称', '证券代码', '报表类别', '截止日期', '公告日期', '基本每股收益', '稀释每股收益']},

            'eps_bank': {'table': 'hk4022',
                         'cols': ['证券简称', '证券代码', '报表类别', '截止日期', '公告日期', '基本每股收益', '稀释每股收益']},

            'kpi': {'table': 'hk4025',
                    'cols': ['证券代码', '截止日期', '报表类别', '毛利率', '净利率', '平均ROE', '资产负债比']},

            'industry': {'table': 'hk4001',
                         'cols': ['证券简称', '证券代码', '一级行业名称', '二级行业名称']}

        }

        if market == 'A':
            self._query_dict = dict_a
        else:
            self._query_dict = dict_h

    def income(self):
        return self._basic_data(dict_key='income')

    def cost(self):
        return self._basic_data(dict_key='cost')

    def efficiency(self):
        return self._basic_data(dict_key='efficiency')

    def warren(self):
        return self._basic_data(dict_key='warren')

    def kpi(self):
        return self._basic_data(dict_key='kpi')

    def bubble(self):
        config = self._query_dict['bubble']
        table, cols = config['table'], config['cols']
        bubble_df = self._db.select(table=table, columns=cols).where(seccodes=self.seccodes, last_year_only=True).sort()
        return bubble_df

    def raw_eps(self):

        if self._market == 'A':
            eps_df = self._basic_data(dict_key='eps', annual=False)
        else:
            eps_df_list = [self._basic_data(dict_key=key, annual=False) for key in ['eps', 'eps_bank']]
            non_empty_dfs = [df for df in eps_df_list if not df.empty]      # 空的df会在下一行产生warning，所以这里过滤掉
            eps_df = pd.concat(non_empty_dfs, ignore_index=True)

        eps_df['稀释每股收益'] = eps_df['稀释每股收益'].combine_first(eps_df['基本每股收益'])
        eps_df.dropna(inplace=True)

        return eps_df

    def price(self, history=True):

        if self._market == 'A':
            table = 'stock2402' if history else 'stock2401'
        else:
            table = 'hk4026'

        price_df = self._db.select(table, columns=['证券代码', '交易日期', '收盘价']).where(self.seccodes, False).sort()

        # 因为停牌，有时候价格会是nan，需要去掉
        price_df.dropna(inplace=True)

        if not history:
            price_df = price_df.groupby('证券代码', sort=False).tail(1)

        return price_df

    def location(self, industry_2):

        # 功能：输入A股二级行业名称，输出这个行业包含的证券简称、代码、城市、三级行业名称、营业收入
        query_str = f'''

        SELECT s21.SECNAME, s21.SECCODE, s21.F028V, s21.F038V, s23.F006N 

            FROM stock2100 s21 INNER JOIN stock2301 s23 ON s21.SECCODE = s23.SECCODE 

            WHERE 
                s21.F036V = "{industry_2}" and 
                s21.CHANGE_CODE <> 2 and 
                LEFT(s21.SECCODE, 1) IN ("0","1","2","3","4","5","6","7","9") and
                s23.F001D = "{newest_fiscal_year()}-12-31" and 
                s23.F002V = "071001" and 
                s23.CHANGE_CODE <> 2;'''

        location_df = self._db.query(query_str, 'stock2100').rename(columns={'F006N': '营业收入'}).dropna()

        return location_df

    def _basic_data(self, dict_key, annual=True):
        # 凡是针对特定一些公司查询指标的，就用这个函数。此函数支持查一个数据库(table)或者查两个数据库(table_1, table_2)再合并

        def q_seccode(t, c):
            return self._db.select(table=t, columns=c).where(seccodes=self.seccodes, annual_only=annual).sort(by=self._year_col, sort_seccode=True)

        config = self._query_dict[dict_key]

        try:
            df = q_seccode(config['table'], config['cols'])

        except KeyError:
            df1 = q_seccode(config['table_1'], config['cols_1'])
            df2 = q_seccode(config['table_2'], config['cols_2'])
            df = df1.merge(df2)

        # 在这里纠正港股年报未必在12月31日的问题
        if '报表类别' in df and not df.empty:
            df['截止日期'] = df.apply(adjust_hk_enddate, axis=1)

        return df


class Segment(object):

    def __init__(self):

        self._db = FinancialDatabase()

        # 获取所有公司的seccode -> 总收入字典，为排序做准备
        income_a = self._whole_db_table('stock2301', ['证券代码', '营业总收入'])
        income_h = self._whole_db_table('hk4024', ['证券代码', '营业额'])
        income_h_bank = self._whole_db_table('hk4021', ['证券代码', '营业收益总额'])

        income_dicts = [df.set_index('证券代码').squeeze().to_dict() for df in [income_a, income_h, income_h_bank]]
        self._di = ChainMap(*income_dicts)

        # 从数据库里获取行业数据的全量df，包括A股的和H股的。
        table, cols = 'stock2100', ['证券简称', '证券代码', '所属城市', '一级行业名称', '二级行业名称', '三级行业名称']
        df_a = self._whole_db_table(table, cols, last_year_only=False)

        table, cols = 'hk4001', ['证券简称', '证券代码', '一级行业名称', '二级行业名称']
        df_h = self._whole_db_table(table, cols, last_year_only=False)

        # 生成A股和H股的全量股票列表和一级行业列表
        self._secnames_a = df_a['证券简称'].unique().tolist()
        self._secnames_h = df_h['证券简称'].unique().tolist()
        self._seccodes_a = df_a['证券代码'].unique().tolist()
        self._seccodes_h = df_h['证券代码'].unique().tolist()

        self._industry_list_a = len_pinyin_sort(df_a['一级行业名称'].unique().tolist())
        self._industry_list_h = len_pinyin_sort(df_h['一级行业名称'].unique().tolist())

        # A股、H股的二、三级行业字典。
        # d代表dict，1、2、3代表一、二、三级行业，c代表code（上市公司代码），n代表name（上市公司名称）
        self._d12_a = include_dict(df_a, '一级行业名称', '二级行业名称')  # 特定 申万一级行业 包含的 二级行业
        self._d23_a = include_dict(df_a, '二级行业名称', '三级行业名称')  # 特定 申万二级行业 包含的 三级行业
        d3c = include_dict(df_a, '三级行业名称', '证券代码')
        self._d3c_a = {k: self._sort(codes) for k, codes in d3c.items()}  # 特定 申万三级行业 包含的 上市公司，已排序

        self._d12_h = include_dict(df_h, '一级行业名称', '二级行业名称')  # 特定 一级行业 包含的 二级行业
        d2c = include_dict(df_h, '二级行业名称', '证券代码')
        self._d2c_h = {k: self._sort(codes) for k, codes in d2c.items()}  # 特定 二级行业 包含的 上市公司，已排序

        self._dnc_a = df_a.set_index('证券简称')['证券代码'].to_dict()  # 特定名称对应的公司代码
        self._dnc_h = df_h.set_index('证券简称')['证券代码'].to_dict()  # 特定名称对应的公司代码

        # 特定代码对应的行业和公司名称。A股、H股的代码没有重叠性，所以两个字典可以融合为1个。
        df_a.set_index('证券代码', inplace=True)
        self._dc1_a = df_a['一级行业名称'].to_dict()
        self._dc2_a = df_a['二级行业名称'].to_dict()
        self._dc3_a = df_a['三级行业名称'].to_dict()

        df_h.set_index('证券代码', inplace=True)
        self._dc1_h = df_h['一级行业名称'].to_dict()
        self._dc2_h = df_h['二级行业名称'].to_dict()

        self._dcn = ChainMap(df_a['证券简称'].to_dict(), df_h['证券简称'].to_dict())

        # 把这个df永久保留在instance里，服务于画map图
        self._df_a = df_a.reset_index()

        # 读取城市GPS坐标数据，服务于画map图
        geo_config = toml.load('configuration.toml')['geo']
        self._geo_df = pd.read_excel(geo_config['geo_file_path'])


    def stocks(self, market):
        return self._secnames_a if market == 'A' else self._secnames_h

    def industry_list(self, market):
        return self._industry_list_a if market == 'A' else self._industry_list_h

    def code(self, secname, market=None):

        # 这个函数支持两种场景：
        # 1. GUI界面下，用户会选择确定的secname和market，因此这时函数输出单一的seccode
        # 2. CLI界面下，用户只输入secname（或secname的一部分），这时有可能出现A股、港股都有的情况，因此这时输出一个list

        if market == 'A':
            return self._dnc_a.get(secname)

        if market == 'H':
            return self._dnc_h.get(secname)

        if not market:
            code_list_a = [code for name, code in self._dnc_a.items() if secname in name]   # 模糊搜索，关键字在secname里即可
            code_list_h = [code for name, code in self._dnc_h.items() if secname in name]
            results = code_list_a + code_list_h
            return [item for item in results if item]

    def name(self, seccode):
        return self._dcn.get(seccode)

    def belong(self, seccode):

        if self.market_a(seccode):
            return self._dc1_a[seccode], self._dc2_a[seccode], self._dc3_a[seccode]

        if self.market_h(seccode):
            return self._dc1_h[seccode], self._dc2_h[seccode], None

    def d12(self, industry, market):
        if market == 'A':
            return self._d12_a[industry]
        else:
            return self._d12_h[industry]

    def d23(self, industry, market):
        if market == 'A':
            return self._d23_a[industry]
        else:
            return []

    def d3c(self, level_2_industry, level_3_industry, market):
        if market == 'A':
            return self._d3c_a[level_3_industry]
        else:
            return self._d2c_h[level_2_industry]  # 注意：为了统一外部调用名称，函数叫做d3c。但港股实际是用d2c字典来查询的，因为港股没有三级行业。

    def peers(self, seccode):

        if self.market_a(seccode):
            industry_3 = self._dc3_a[seccode]
            return self._d3c_a[industry_3]

        if self.market_h(seccode):
            industry_2 = self._dc2_h[seccode]
            return self._d2c_h[industry_2]

    def market_a(self, seccode):
        return seccode in self._seccodes_a

    def market_h(self, seccode):
        return seccode in self._seccodes_h

    def location(self, industry_2):
        # 输入二级行业名称，输出此行业包含公司的所在城市、GPS坐标、去年总收入

        # 过滤出特定二级行业的公司名称。为了避免修改原变量，做了copy操作。
        location_df = self._df_a[self._df_a['二级行业名称'] == industry_2].copy()

        location_df['总收入'] = location_df['证券代码'].map(self._di)
        merged_df = location_df.merge(self._geo_df, left_on='所属城市', right_on='区域名称')

        return merged_df

    def _sort(self, seccodes):
        return sorted(seccodes, key=lambda code: self._di[code] if code in self._di else 0, reverse=True)

    def _whole_db_table(self, table, columns, last_year_only=True):
        return self._db.select(table, columns).where(seccodes=None, annual_only=False, last_year_only=last_year_only).sort()


class FinancialDatabase(object):

    def __init__(self):

        # 加载配置
        config = toml.load('configuration.toml')
        db = config['database']

        # 创建数据库 engine
        conn_string = f"mysql+pymysql://{db['account']}:{db['password']}@{db['address']}:{db['port']}/{db['name']}"
        self._engine = create_engine(conn_string)

        # 数据库名称字典
        dfs = pd.read_excel(db['name_dictionary_path'], sheet_name=None)
        del dfs['说明']

        self._db2real_dicts = {sheet: df.set_index('数据库名称')['中文名称'].squeeze().to_dict() for (sheet, df) in
                               dfs.items()}
        self._real2db_dicts = {sheet: df.set_index('中文名称')['数据库名称'].squeeze().to_dict() for (sheet, df) in
                               dfs.items()}

        # 关于数据库的一些补充规则，具体原因见name_dictionary.xlsx的“说明”sheet
        self._table_specific_rule = {
            'stock2100': 'LEFT(SECCODE, 1) IN ("0","1","2","3","4","5","6","7","9") and CHANGE_CODE in (1, 3) and F034V is not null',  # 排除8开头code
            'stock2301': 'F002V = "071001" and CHANGE_CODE in (1,3)',
            'stock2302': 'F002V = "071001" and CHANGE_CODE in (1,3)',
            'stock2303': 'F070V = "071001" and CHANGE_CODE in (1,3) and F001V = "033003"',
            'stock2401': '',
            'stock2402': '',
            'hk4001': '',
            'hk4021': '',
            'hk4022': '',
            'hk4023': '',
            'hk4024': '',
            'hk4025': '',
            'hk4026': ''}

        self._table = ''
        self._seccodes = []

        self._select_str = ''
        self._where_str = ''
        self._sort_str = ''

    def select(self, table, columns):

        # 这个select函数是生成query语句的开始。因此这个位置对生成一个完整query string所需的几个隐藏变量进行初始化
        # 也就是说，一旦你开始调用select()，你就要从头开始构建这个query string
        self._select_str = ''
        self._where_str = ''
        self._sort_str = ''

        self._table = table
        column_str = ', '.join(self._get_db_name(table, columns))
        self._select_str = f'select {column_str} from {table}'

        return self

    def where(self, seccodes, annual_only=True, last_year_only=False):

        code_rule = annual_rule = last_year_rule = None

        if seccodes:
            self._seccodes = seccodes
            code_str = ', '.join(double_quote(seccodes))
            code_rule = f'SECCODE in ({code_str})'

        if annual_only and self._table in ['stock2300', 'stock2301', 'stock2302', 'stock2303']:
            time_name = self._get_db_name(self._table, '截止日期')
            annual_rule = f'{time_name} LIKE "____-12-31"'      # 曾用方法：REGEXP "12-31"

        if annual_only and self._table in ['hk4021', 'hk4022', 'hk4023', 'hk4024', 'hk4025']:
            type_name = self._get_db_name(self._table, '报表类别')
            annual_rule = f'{type_name} = "年报"'

        if last_year_only:
            time_name = self._get_db_name(self._table, '截止日期')
            last_year_rule = f'{time_name} = "{newest_fiscal_year()}-12-31"'

        other_rules = self._table_specific_rule[self._table]
        where_rules = [code_rule, annual_rule, last_year_rule, other_rules]

        self._where_str = multi_rules(prefix='where', connector=' and ', rule_list=where_rules)

        return self

    def sort(self, by='', sort_seccode=False):

        by_rule = case_rule = None

        if by:
            by_db_name = self._get_db_name(self._table, by)
            by_rule = f'{by_db_name} desc'

        if sort_seccode:
            # case_list形成 ['when "招商银行" then 1', 'when "格力电器" then 2']
            when_list = [f'when {item} then {i}' for i, item in enumerate(double_quote(self._seccodes), start=1)]
            when_str = ' '.join(when_list)
            case_rule = f'case SECCODE {when_str} end'

        self._sort_str = multi_rules(prefix='order by', connector=', ', rule_list=[by_rule, case_rule])

        sql_string = f'{self._select_str} {self._where_str} {self._sort_str}'.strip() + ';'
        return self.query(sql_string, self._table)

    def query(self, query_string, table_name):
        df = pd.read_sql_query(query_string, self._engine, parse_dates=['ENDDATE', 'TRADEDATE', 'DECLAREDATE'])
        return df.rename(columns=self._db2real_dicts[table_name])

    def _get_db_name(self, table_name, name):

        table_dict = self._real2db_dicts[table_name]

        if isinstance(name, list):
            return [table_dict[item] for item in name if item in table_dict]
        elif isinstance(name, str):
            return table_dict[name] if name in table_dict else None
        else:
            return None


def adjust_hk_enddate(row):

    date = row['截止日期']
    q2_date_is_right = (row['报表类别'] == '半年报' and date.month == 6)
    q4_date_is_right = (row['报表类别'] == '年报' and date.month == 12)

    if q2_date_is_right or q4_date_is_right:
        return date

    this_year = date.year
    last_year = this_year - 1

    if row['报表类别'] == '半年报':
        return datetime(last_year if date.month < 6 else this_year, 6, 30)
    else:
        return datetime(last_year, 12, 31)      # 如果不是'半年报'，则肯定是'年报'


def double_quote(list_or_string):
    # 这个函数在一个或多个元素上添加双引号 ""，为生成mySQL的query语句做准备。

    # 若输入的是list则直接用。若不是，外面套一个[]变成list
    work_list = list_or_string if isinstance(list_or_string, list) else [list_or_string]

    # 给work_list里的element加上双引号。例如从 ['招商银行', '格力电器']变成了['"招商银行"', '"格力电器"']
    double_quote_list = [f'"{item}"' for item in work_list]

    return double_quote_list


def multi_rules(prefix, connector, rule_list):
    actual_list = [item for item in rule_list if item]
    result_string = f'{prefix} {connector.join(actual_list)}' if actual_list else ''
    return result_string


def newest_fiscal_year():
    # 4月以后，去年年报都出来了，因此newest year是去年；反之，newest year是前年
    current_year, current_month = datetime.today().year, datetime.today().month
    newest_year = current_year - 1 if current_month > 4 else current_year - 2

    return newest_year


def eps_at_date(eps_series, a_date):
    current_eps = eps_series.loc[a_date]

    if a_date.month == 12:
        static = ttm = forcast = current_eps
    else:
        try:
            previous_year_end = a_date - pd.offsets.YearEnd(1)
            static = eps_series.loc[previous_year_end]

            one_year_ago_eps = eps_series.loc[datetime(a_date.year - 1, a_date.month, a_date.day)]
            ttm = static - one_year_ago_eps + current_eps

            forcast = None if one_year_ago_eps == 0.0 else (current_eps / one_year_ago_eps) * static

        except KeyError:
            static = ttm = forcast = None

    return {'eps_static': static, 'eps_ttm': ttm, 'eps_forcast': forcast}


def n_year_mean(a_series, past_n_year):
    _, end_date = first_last_year(a_series)
    start_date = end_date - pd.offsets.YearEnd(past_n_year - 1)
    target_series = a_series.loc[end_date: start_date]

    return target_series.mean() if len(target_series) == past_n_year else np.nan


def first_last_year(data_series):

    latest_date = data_series.index.max()
    last_year = latest_date if latest_date.month == 12 else latest_date - pd.offsets.YearEnd(1)

    first_year = last_year

    if last_year not in data_series.index:
        return first_year, last_year

    while True:
        previous = first_year - pd.offsets.YearEnd(1)
        if previous in data_series:
            first_year = previous
        else:
            return first_year, last_year


def growth_rate(data_series, past_n_year=None):
    data_series = data_series.dropna()

    if data_series.empty:
        return np.nan

    # 删除eps为0的数据，解决当 first_year = 0 时没法计算成长率的问题
    non_zero_mask = data_series != 0
    data_series = data_series[non_zero_mask]

    first_year, last_year = first_last_year(data_series)
    if past_n_year:
        first_year = last_year - pd.offsets.YearEnd(past_n_year)
    else:
        past_n_year = last_year.year - first_year.year
        if past_n_year == 0:
            return np.nan

    try:
        first_year_data = data_series.loc[first_year]
        last_year_data = data_series.loc[last_year]
    except KeyError:
        return np.nan

    # 当期初、期末数据都为正数(例如3年数据为2, 4, 8)：采用最普通的公式
    if first_year_data > 0 and last_year_data > 0:

        compound_growth_rate = (last_year_data / first_year_data) ** (1 / past_n_year) - 1

    # 当期初为负，但期末为正(例如3年数据为-2, 0, 4)：将期初值平移到正数对应值（-2平移到2），所有值都平移相同数量，然后应用全正公式
    elif first_year_data < 0 < last_year_data:

        shift_value = 2 * abs(first_year_data)

        first_year_data += shift_value
        last_year_data += shift_value

        compound_growth_rate = (last_year_data / first_year_data) ** (1 / past_n_year) - 1

    # 当期初为正，但期末为负(例如3年数据为2, 0, -4)：将期初值平移到负数对应值（2平移到-2），所有值都平移相同数量，然后应用全负公式
    elif first_year_data > 0 > last_year_data:

        shift_value = -2 * abs(first_year_data)

        first_year_data += shift_value
        last_year_data += shift_value

        compound_growth_rate = (last_year_data / first_year_data) ** (1 / past_n_year) - 1
        compound_growth_rate = -compound_growth_rate

    # 剩下的场景就是期初、期末数据都为负数时(例如3年数据为-2, -4, -8)。这时采用全负公式：先按全正公式计算，然后在结果上增加一个负号
    else:

        compound_growth_rate = (last_year_data / first_year_data) ** (1 / past_n_year) - 1
        compound_growth_rate = -compound_growth_rate

    compound_growth_rate = compound_growth_rate * 100

    return compound_growth_rate


def to_str(item, decimal=1):
    return f'{item}' if isinstance(item, (str, int)) else f'{item:.{decimal}f}'


def len_pinyin_sort(a_list):
    # 这个函数用来对一个list中的元素排序。首先考虑元素长度，其次，相同长度时考虑拼音顺序。
    # 港股的行业列表里有可能会出现None（可能是还没分配行业的待上市公司），因此先去除None
    a_list = [x for x in a_list if x is not None]
    return sorted(a_list, key=lambda x: (len(x), slug(x)))


def include_dict(df, key_column, value_column):
    return df.groupby(key_column)[value_column].apply(set).apply(list).apply(len_pinyin_sort).to_dict()

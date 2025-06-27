import re
from re import search

import toml
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from datetime import datetime
from plotly.subplots import make_subplots

from rich import box
from rich.live import Live
from rich.text import Text
from rich.tree import Tree
from rich.table import Table
from rich.theme import Theme
from rich.style import Style
from rich.console import Console
from rich.columns import Columns
from rich.markdown import Markdown
from rich.progress import Progress

from itertools import chain
from functools import partial
from collections import ChainMap
from core.database import Data, SingleProcessor, Segment


class WebFig(object):

    def __init__(self, seccodes, market):
        self._seccodes = seccodes
        self._market = market
        self._data = Data(seccodes, market)

    def income_fig(self, title):

        df = self._data.income()

        if self._market == 'A':
            y_cols = ['营业收入', '营业利润', '利润总额', '净利润', '归母净利润', '扣非净利润', '经营现金流']
        else:
            y_cols = ['营业额', '经营溢利', '除税前经营溢利', '除税后经营溢利', '股东应占溢利', '经营现金流']

        return bar(df, y_cols, x_title='', y_title='单位：亿', title=title, barmode='group', yi=True)

    def cost_fig(self, title):

        if self._market == 'A':
            title_col = ['截止日期', '证券简称']
            revenue_col = ['营业总收入']
            subtract_col = ['财务费用', '研发费用', '管理费用', '销售费用', '营业税', '利息支出', '营业成本']
            supplement_col = ['投资收益', '公允价值变动收益', '信用减值损失（2019格式）', '资产减值损失（2019格式）',
                              '资产减值损失', '资产处置收益', '其它收入']
        else:
            title_col = ['截止日期', '证券简称']
            revenue_col = ['营业额']
            subtract_col = ['财务成本', '行政费用', '销售及分销成本', '折旧', '贷款利息', '经营开支总额']
            supplement_col = ['特殊项目', '联营公司']

        df = self._data.cost().set_index(title_col)

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

        y_cols = percentage_df.columns

        fig = bar(percentage_df.reset_index(), y_cols, x_title='', y_title='', title=title, barmode='relative')
        fig.layout.update(yaxis=dict(tickformat='.0%'))

        return fig

    def efficiency_fig(self, title):

        y_cols_a = ['毛利率', '营业利润率', '净利率', 'ROA', '加权平均ROE']
        y_cols_h = ['毛利率', '税前利润率', '净利率', '平均ROA', '平均ROE']

        df = self._data.efficiency()
        y_cols = y_cols_a if self._market == 'A' else y_cols_h

        return bar(df, y_cols, x_title='', y_title='', title=title, barmode='group')

    def warren_fig(self, title):

        df = self._data.warren().dropna(axis=0, how='all').dropna(axis=1, how='all').set_index('截止日期')
        df = df.drop(columns='报表类别') if '报表类别' in df else df
        y_cols = df.columns
        df.reset_index(inplace=True)

        return area(df, y_cols, x_title='', y_title='', title=title)

    def valuation_fig(self, title):

        # 获取估值数据
        eps, price = self._data.raw_eps(), self._data.price()
        pe_df = SingleProcessor(seccode=self._seccodes, raw_eps_df=eps, raw_price_df=price).pe()
        pe_df = pe_df.set_index('交易日期')[['eps_ttm', 'pe_ttm', '收盘价']]

        start_date, end_date = pe_df.index.min(), pe_df.index.max()
        total_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)

        # 原始的 pe_df 的颗粒度是工作日，但如果公司存在时间很长，则按照年度重新采样更平滑、好看。
        # A-May 表示年度采样而且以May做为每年的开头。这样的好处在于五月份刚刚把所有的年报更新完，五月的价格考虑了最新业绩。
        if total_months > 120:
            chart_df = pe_df.resample('A-MAY', convention='end').nearest().loc[: datetime.today()]
        elif 24 < total_months < 120:
            chart_df = pe_df.resample('BQ', convention='end').nearest().loc[: datetime.today()]
        else:
            chart_df = pe_df.resample('BM', convention='end').nearest().loc[: datetime.today()]

        chart_df = chart_df.round({'eps_ttm': 1, 'pe_ttm': 0, '收盘价': 1})

        # 画图。尝试过画在一起但大小差异很大，不好看。因此分成三个独立的图。
        fig = make_subplots(rows=3, cols=1, subplot_titles=['每股收益（TTM）', '市盈率（TTM）', '股价'],
                            vertical_spacing=0.1)

        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['eps_ttm'], mode='lines+markers+text',
                                 text=chart_df['eps_ttm'], textposition='top center'), row=1, col=1)
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['pe_ttm'], mode='lines+markers+text',
                                 text=chart_df['pe_ttm'].round(), textposition='top center'), row=2, col=1)
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['收盘价'], mode='lines+markers+text',
                                 text=chart_df['收盘价'].round(1), textposition='top center'), row=3, col=1)

        max_width = 1400
        look_good_width = len(chart_df) * 60

        fig.update_layout(height=800, width=min(look_good_width, max_width), title_text=title, showlegend=False)

        return fig

    def gui_table(self):

        eps, price, kpi = self._data.raw_eps(), self._data.price(history=False), self._data.kpi()

        processor_list = [SingleProcessor(code, eps, price, kpi) for code in self._seccodes]
        summary_list = [processor.summary_line() for processor in processor_list]

        table_data = [self._table_data_row(summary_line) for summary_line in summary_list]
        tooltip_data = [self._table_tooltip_row(summary_line) for summary_line in summary_list]

        return table_data, tooltip_data

    def industry_map(self, industry_2_name):

        # 如果不是A股，则不画图。因为只有A股有数据。
        if self._market != 'A':
            return None

        # 上市公司所在的城市
        location_df = self._data.location(industry_2_name)

        # 附加这些城市的GPS位置
        geo_config = toml.load('../configuration.toml')['geo']

        px.set_mapbox_access_token(geo_config['mapbox_token'])
        geo_df = pd.read_excel(geo_config['geo_file_path'])

        merged_df = location_df.merge(geo_df, left_on='所属城市', right_on='区域名称')

        # 整理数据
        merged_df['城市+公司'] = merged_df['所属城市'].str.cat(merged_df['证券简称'], sep='<br>')

        # 根据总收入，划分几个等级，体现为图里的size参数，即bubble大小
        range_bins = [0, 1.0e8, 1.0e9, 1.0e10, 1.0e11, 1.0e13]
        label_list = [1, 2, 6, 12, 20]
        merged_df['收入级别'] = pd.cut(merged_df['营业收入'], bins=range_bins, labels=label_list, include_lowest=True)

        # 画图
        title = f'申万二级行业：{industry_2_name}'
        fig = px.scatter_mapbox(data_frame=merged_df, title=title, lat='latitude', lon='longitude',
                                text='城市+公司', hover_name='证券简称', color='三级行业名称', size='收入级别',
                                size_max=20,
                                zoom=4, opacity=0.5, width=1400, height=1000, center=dict(lat=35, lon=105))

        fig.layout.update(title=dict(x=0.5))

        return fig

    def bubble_chart(self, industry_2_name):

        # 获取估值数据
        eps, price = self._data.raw_eps(), self._data.price(history=False)
        pe_df = SingleProcessor(seccode=self._seccodes, raw_eps_df=eps, raw_price_df=price).pe()
        pe_df = pe_df.set_index('交易日期')[['eps_ttm', 'pe_ttm', '收盘价']]

        return

    @staticmethod
    def _table_data_row(summary_table_line):

        # 初步处理一下
        row = summary_table_line.reset_index().squeeze()

        data_row = {'公司': markdown_cell(text=row['证券简称'], link=report_link(row['证券代码'])),
                    '上市年限': row['上市年限'],
                    '价格': row['收盘价'],
                    '毛利率': multi_lines(['1年', '5年', '10年'], row['毛利率'].split()),
                    '净利率': multi_lines(['1年', '5年', '10年'], row['净利率'].split()),
                    'ROE': multi_lines(['1年', '5年', '10年'], row['ROE'].split()),
                    '资产负债比': multi_lines(['1年', '5年', '10年'], row['资产负债比'].split()),
                    'PE': multi_lines(['静态', 'TTM', '预测'], [row.pe_static, row.pe_ttm, row.pe_forcast]),
                    'EPS成长性': multi_lines(['3年', '5年', '10年'], row['EPS成长性'].split()),
                    '估值gap': row['估值gap'],
                    '总收益': row['总收益']}

        return data_row

    @staticmethod
    def _table_tooltip_row(summary_table_line):

        # 初步处理一下
        row = summary_table_line.reset_index().squeeze()

        prefix_1 = ['静态EPS', 'TTM EPS', '预测 EPS']
        eps_list = [row.eps_static, row.eps_ttm, row.eps_forcast]

        tooltip_row = {'PE': {'value': multi_lines(prefix_1, eps_list, seperator='\\\n'), 'type': 'markdown'},
                       '价格': f'日期：{row["交易日期"].strftime("%Y-%m-%d")}'}

        return tooltip_row


class Cli(object):

    def __init__(self):

        self._seg = Segment()

        # 根据行业字典，生成一个rich tree
        self._generate_tree()

        self._cli_columns = ['一级行业', '二级行业', '三级行业',
                             '证券简称', '链接', '上市年限',
                             '毛利率', '净利率', 'ROE', '资产负债比',
                             'PE(静态 动态 预测)', 'EPS成长性', '估值gap', '总收益']

        # 设置不同指标的评价体系。注意：只有 资产负债比 越大越不好，因此用了decreasing_labels。
        increasing_labels = ['bad', 'fair', 'good', 'great']
        decreasing_labels = ['great', 'good', 'fair', 'bad']
        self._kpi_dict = {
            '毛利率': {'bins': [0, 20, 50, 75, 1000], 'labels': increasing_labels},
            '净利率': {'bins': [0, 10, 15, 30, 1000], 'labels': increasing_labels},
            'ROE': {'bins': [0, 10, 15, 30, 1000], 'labels': increasing_labels},
            'EPS成长性': {'bins': [0, 5, 10, 20, 1000], 'labels': increasing_labels},
            '资产负债比': {'bins': [0, 10, 20, 50, 1000], 'labels': decreasing_labels},
            '估值gap': {'bins': [0, 5, 10, 20, 1000], 'labels': increasing_labels},
            '总收益': {'bins': [0, 10, 20, 50, 1000], 'labels': increasing_labels}
        }

        display_styles = {'dark_gold': 'bold #b8860b',
                          'default': Style(dim=True),
                          'bad': Style(dim=True),
                          'fair': Style(dim=True),
                          'good': 'gold1',
                          'great': 'dark_orange',
                          'highlight': Style(bgcolor='grey27', bold=True)}

        self.console = Console(theme=Theme(display_styles))

    def print_tree(self):
        self.console.print(self._tree)
        self.console.print()

    def scan(self):

        industry_2 = '物流'
        stocks = self._codes_2(industry_2, market='A')

        with Progress() as p:
            task1 = p.add_task("[red]Downloading...", total=len(stocks))

            while not p.finished:
                self._output()

        return

    def analyze(self, user_input):

        # 先去掉可能出现的空格
        user_input = user_input.strip()

        # 用户输入A股三级行业代码
        if user_input in self._dict3_a:
            industry_name = self._dict3_a[user_input]
            seccodes = self._codes_3(industry_name, market='A')

        # 用户输入港股二级行业代码
        elif user_input in self._dict2_h:
            industry_name = self._dict2_h[user_input]
            seccodes = self._codes_3(industry_name, market='H')

        # 用户输入A股二级行业代码
        elif user_input in self._dict2_a:
            industry_name = self._dict2_a[user_input]
            seccodes = self._codes_2(industry_name, market='A')

        # 用户输入港股一级行业代码
        elif user_input in self._dict1_h:
            industry_name = self._dict1_h[user_input]
            seccodes = self._codes_2(industry_name, market='H')

        # 用户输入A股一级行业代码
        elif user_input in self._dict1_a:
            industry_name = self._dict1_a[user_input]
            seccodes = self._codes_1(industry_name)

        else:
            # 用户输入了一个或多个公司名称
            seccodes = self._str_to_stocks(user_input)
            if seccodes:
                industry_name = '自定义组合'
            else:
                self.console.print('未找到对应公司，请重新输入')
                return

        report_table = self._get_table(title=industry_name)
        self._output(rich_table=report_table, seccodes=seccodes)

    def _str_to_stocks(self, secname_str):

        # 当用户输入一个或多个股票名字，这时就区分处理。

        # 先看看到底是几个seccode。默认情况下，凑出下面的code_list就算完事了。也就是说只展示这几个公司的数据，不展示它们每个对应的同行。
        secname_list = re.split(pattern=r'\W+', string=secname_str)
        nested_code_list = [self._seg.code(name) for name in secname_list]
        code_list = list(chain.from_iterable(nested_code_list))

        # 要是只有一个名字而且经查确实有seccode，这时额外处理一下：因为这种情况数据不多，所以就把这个股票的同行也展示出来
        if len(secname_list) == 1 and 0 < len(code_list) < 3:
            nested_code_list = [self._seg.peers(code) for code in code_list]
            code_list = list(chain.from_iterable(nested_code_list))

        return code_list

    def _output(self, rich_table, seccodes):

        # 根据seccodes触发一次数据获取
        processor_dict = batch_process(seccodes)

        with Live(rich_table, refresh_per_second=100, console=self.console, vertical_overflow='ellipsis'):

            for seccode in seccodes:
                rendered_row, should_highlight = self._cli_line(seccode, processor_dict[seccode].summary_line())
                rich_table.add_row(*rendered_row.values(), style='highlight' if should_highlight else None)

    def _get_table(self, title):

        # 生成rich table的表头，并设置column宽度
        cap_text = r'三个数字分别表示：1年 5年 10年 算术平均     EPS成长性：3年 5年 10年几何平均'
        table = Table(title=title, box=box.HORIZONTALS, caption=cap_text, caption_justify='right')

        # 证券简称用的是markdown格式，如果不指定宽度（即自适应宽度），就特别长（应该是bug）。考虑美观，在这里进行了限制。
        for column in self._cli_columns:

            if column == '证券简称':
                col_width = 14
            elif column == '链接':
                col_width = 10
            else:
                col_width = None

            table.add_column(header=column, vertical='middle', max_width=col_width)

        return table

    def _cli_line(self, seccode, summary_line):

        # 修补问题：刚上市的公司还没有eps数据，所以eps df是空的，但证券简称来自于那里。所以在这里补充一下。
        if '证券简称' not in summary_line:
            summary_line['证券简称'] = self._seg.name(seccode)

        if summary_line.empty:
            cli_row = self._abnormal_line(seccode, '未上市')
        else:
            # 基于summary_line进一步加工成需要的Cli_line形式
            cli_line = summary_line.reset_index().rename(columns={'PE': 'PE(静态 动态 预测)'})

            # 补充行业信息
            cli_line[['一级行业', '二级行业', '三级行业']] = self._seg.belong(seccode)

            # 提供“年报”和“分红”的超链接
            report = markdown_cell(text='年报', link=report_link(seccode))
            dividend = markdown_cell(text='分红', link=dividend_link(seccode))
            cli_line['链接'] = Markdown(report + ' ' + dividend)

            # 上面数据处理完毕后，这里对这行数据进行渲染涂色
            cli_row = cli_line[self._cli_columns].squeeze()

        rendered_row = {kpi_name: self._render_value(kpi_name, kpi_value) for kpi_name, kpi_value in cli_row.items()}

        # 针对这一行数据，检查一下是否属于good business和good price
        is_good_business = over_threshold(rendered_row, ['毛利率', '净利率', 'ROE'], 4)
        is_good_price = over_threshold(rendered_row, ['总收益'])

        # 根据评价结果，判断是否高亮这行
        highlight = True if (is_good_business and is_good_price) else False

        return rendered_row, highlight

    def _abnormal_line(self, seccode, error_text):

        # i1, i2, i3 = self._seg.belong(seccode)
        # name = self._seg.name(seccode)
        error_dict = {col: error_text for col in self._cli_columns}
        output_series = pd.Series(error_dict)

        output_series.loc[['一级行业', '二级行业', '三级行业']] = [self._seg.belong(seccode)]
        output_series.loc['证券简称'] = self._seg.name(seccode)

        # # 通过字典创建Series
        # data = {'一级行业': i1,
        #         '二级行业': i2,
        #         '三级行业': i3,
        #         '证券简称': name,
        #         '链接': error_text,
        #         '上市年限': error_text,
        #         '毛利率': error_text,
        #         '净利率': error_text,
        #         'ROE': error_text,
        #         '资产负债比': error_text,
        #         'PE(静态 动态 预测)': error_text,
        #         'EPS成长性': error_text,
        #         '估值gap': error_text,
        #         '总收益': error_text}

        return output_series

    def _codes_3(self, industry_name, market):
        return self._seg.d3c(level_2_industry=industry_name, level_3_industry=industry_name, market=market)

    def _codes_2(self, industry_name, market):
        industry_names = self._seg.d23(industry_name, market) if market == 'A' else self._seg.d12(industry_name, market)
        nested_code_list = [self._codes_3(name, market) for name in industry_names]
        return list(chain.from_iterable(nested_code_list))

    def _codes_1(self, industry_name, market='A'):
        industry_2_names = self._seg.d12(industry_name, market=market)
        nested_code_list = [self._codes_2(name, market) for name in industry_2_names]
        return list(chain.from_iterable(nested_code_list))

    def _render_value(self, kpi_name, kpi_value):

        # 如果是markdown类型，说明希望按照markdown方式进行渲染，这里就不需要进行后续处理了
        if isinstance(kpi_value, Markdown):
            return kpi_value

        # 如果没找到这个KPI对应的评价字典（eval_dict），说明不需要评价它，直接返回默认色
        if kpi_name in self._kpi_dict:
            eval_dict = self._kpi_dict[kpi_name]
        else:
            return Text(str(kpi_value), 'default')

        # value有时是被空格分隔的几个数字或字符，有时只有一个。先用下面的命令变成一个list。当只有一个数字时，这个list只有一个item。
        values = [to_number(n) for n in kpi_value.split()]
        styles = pd.cut(x=values, bins=eval_dict['bins'], labels=eval_dict['labels']).fillna('bad').tolist()

        rich_text = Text()
        for value, style in zip(values, styles):
            rich_text.append(Text(str(value), style)).pad_right(1)

        return rich_text

    def _generate_tree(self):

        # 生成行业树的根目录。每个行业是一个Rich格式的Tree。trees_a是A股的行业树list， trees_h是港股的行业树list。
        set_a = [(str(i).zfill(2), name) for i, name in enumerate(self._seg.industry_list('A'))]
        set_h = [(str(i).zfill(2), name) for i, name in enumerate(self._seg.industry_list('H'), start=len(set_a))]

        trees_a = [Tree(label=f'{name} {key}', style='dark_gold', guide_style='dark_gold') for key, name in set_a]
        trees_h = [Tree(label=f'{name} {key}', style='great', guide_style='great') for key, name in set_h]

        # 生成A股的第二、第三级的树节点，以及港股的第二级节点。返回值是数字 -> 行业名称的映射表，供以后查询时使用。
        d12_a = partial(self._seg.d12, market='A')
        d23_a = partial(self._seg.d23, market='A')
        d12_h = partial(self._seg.d12, market='H')
        dict_2_list_a = [self._add_branch(d12_a, tree, 'dark_gold', 'dim blue') for tree in trees_a]
        dict_3_list_a = [self._add_branch(d23_a, tree, 'dim blue') for tree in trees_a]
        dict_2_list_h = [self._add_branch(d12_h, tree, style='green4') for tree in trees_h]

        # 把A股和H股的tree合在一起，用column方式呈现
        self._tree = Columns(trees_a + trees_h, padding=4)

        # 生成需要的字典格式
        self._dict1_a = dict((key, name) for key, name in set_a)
        self._dict1_h = dict((key, name) for key, name in set_h)
        self._dict2_a = ChainMap(*dict_2_list_a)
        self._dict3_a = ChainMap(*dict_3_list_a)
        self._dict2_h = ChainMap(*dict_2_list_h)

    @staticmethod
    def _add_branch(mapping_func, node, style=None, guide_style=None):

        number_dict = {}
        for edge_node in get_leaf_nodes(node):

            name, number = edge_node.label.split(' ')
            number = search('[0-9]+', number).group()
            sub_names = mapping_func(name)

            for i, sub_name in enumerate(sub_names):
                sub_number = f'{number}{i}'
                edge_node.add(f'{sub_name} ({sub_number})', style=style, guide_style=guide_style)

                number_dict[sub_number] = sub_name

        return number_dict


class Holdings(object):

    def __init__(self, holding_dictionary):
        self._holdings = holding_dictionary

    def holding_info(self, hkd_rate=0.939):

        code_list = self._holdings.keys()
        needed_cols = ['证券简称', '收盘价', 'PE', 'ROE', '估值gap', '总收益']

        # 获取基本信息（名称、价格、估值、总收益）
        processor_dict = batch_process(seccode_list=code_list)
        summary_line_list = [processor_dict[code].summary_line()[needed_cols] for code in code_list]

        # 补充持仓数量、市值
        amount_df = pd.DataFrame.from_dict(self._holdings, orient='index', columns=['持仓数量'])
        holding_df = pd.concat(summary_line_list).merge(amount_df, left_index=True, right_index=True)
        holding_df['市值'] = holding_df['持仓数量'] * holding_df['收盘价']

        # 考虑汇率因素
        currency_rate_dict = {code: self._currency(code=code, hkd_rmb_rate=hkd_rate) for code in code_list}
        currency_df = pd.DataFrame.from_dict(currency_rate_dict, orient='index', columns=['汇率'])
        holding_df = holding_df.merge(currency_df, left_index=True, right_index=True)
        holding_df['人民币市值'] = holding_df['市值'] * holding_df['汇率']

        # 补充仓位占比
        total_cap = holding_df['人民币市值'].sum()
        holding_df['仓位'] = (holding_df['人民币市值'] / total_cap).round(3)

        market_cap_cols = ['市值', '人民币市值']
        holding_df[market_cap_cols] = (holding_df[market_cap_cols] / 10000).round(1)

        return holding_df

    @staticmethod
    def _currency(code, hkd_rmb_rate):

        if is_a(code):
            return 1

        if is_h(code):
            return hkd_rmb_rate


def batch_process(seccode_list):

    # 将seccodes分为 A股、H股两组。最后形成 {'A':[ a股list ], 'H':[ h股list ]}
    filter_funcs = {'A': is_a, 'H': is_h}
    a_h_subsets = {market: list(filter(func, seccode_list)) for market, func in filter_funcs.items()}

    # 区分A股、H股，进行数据库提取，输出epk三个df(e:eps, p:price, k:kpi)，供后续使用
    data_list = [Data(subset, market) if subset else None for market, subset in a_h_subsets.items()]
    epk_a, epk_h = [(data.raw_eps(), data.price(history=False), data.kpi()) if data else None for data in data_list]

    single_processor_dict = {}
    for code in seccode_list:
        single_processor = SingleProcessor(code, *epk_a) if is_a(code) else SingleProcessor(code, *epk_h)
        single_processor_dict[code] = single_processor

    return single_processor_dict


def is_a(seccode):
    return len(seccode) == 6


def is_h(seccode):
    return len(seccode) == 5


def to_str(item, decimal=1):
    return f'{item}' if isinstance(item, (str, int)) else f'{item:.{decimal}f}'


def to_number(n):
    try:
        return float(n)
    except ValueError:
        return n


def multi_lines(prefix_list, number_list, seperator='\n', decimal=1):
    merged_list = [f'{prefix}:{to_str(number, decimal)}' for prefix, number in zip(prefix_list, number_list)]
    merged_string = seperator.join(merged_list)

    return merged_string


def markdown_cell(text, link):
    return f'''[{text}]({link})'''


def report_link(sec_code):

    if len(sec_code) == 5:
        return f'https://stock.finance.sina.com.cn/hkstock/notice/{sec_code}.html'
    else:
        return f'http://money.finance.sina.com.cn/corp/go.php/vCB_Bulletin/stockid/{sec_code}/page_type/ndbg.phtml'


def dividend_link(sec_code):

    if len(sec_code) == 5:
        code_prefix = ''
    elif str(sec_code).startswith('6'):
        code_prefix = 'SH'
    else:
        code_prefix = 'SZ'

    return f'https://xueqiu.com/snowman/S/{code_prefix}{sec_code}/detail#/FHPS'


def get_leaf_nodes(node):
    if node.children:
        for child in node.children:
            yield from get_leaf_nodes(child)
    else:
        yield node


def over_threshold(rendered_row, kpi_names, threshold_number=1):

    span_list = list(chain.from_iterable([rendered_row[kpi].spans for kpi in kpi_names]))
    style_list = [span.style for span in span_list]

    # over的定义：至少达到 good
    over_number = style_list.count('good') + style_list.count('great')
    if over_number >= threshold_number:
        return True
    else:
        return False


def bar(df, y_cols, x_title, y_title, title, barmode, yi=False):

    year_col = '截止日期'
    name_col = '证券简称'

    # 只显示年份
    df[year_col] = df[year_col].dt.year

    if yi:
        df[y_cols] = df[y_cols].divide(1.0e8).round(1)

    name_count = df[name_col].nunique()
    single = True if name_count == 1 else False

    if single:
        fig = px.bar(data_frame=df, x=year_col, y=y_cols, title=title, opacity=0.8)
    else:
        fig = px.bar(data_frame=df, x=name_col, y=y_cols, animation_frame=year_col, title=title, opacity=0.8)

    fig.layout.update(xaxis_title=x_title, yaxis_title=y_title, barmode=barmode, title=dict(x=0.5))

    return fig


def area(df, y_cols, x_title, y_title, title):

    year_col = '截止日期'

    # 只显示年份
    df[year_col] = df[year_col].dt.year

    fig = px.area(data_frame=df, x=year_col, y=y_cols, title=title)
    fig.layout.update(xaxis_title=x_title, yaxis_title=y_title, title=dict(x=0.5))

    return fig


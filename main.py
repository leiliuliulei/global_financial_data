import toml
import dash
import pandas as pd
from random import randint
from datetime import datetime
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sqlalchemy import create_engine
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
from db_functions import MySQLQuery, income_chart, cost_chart, efficiency_chart, \
    warren_touch_chart, map_figure, annual_report_link, actual_growth, suggested_growth, add_eps_pe

# 加载配置
config = toml.load('configuration.toml')

# 获得数据库配置
db = config['database']
connection_string = 'mysql+pymysql://{account}:{password}@{address}:{port}/{name}'.format(
    account=db['account'],
    password=db['password'],
    address=db['address'],
    port=db['port'],
    name=db['name']
)
financial_db_engine = create_engine(connection_string)
db_query = MySQLQuery(financial_db_engine, db['dictionary_file_path'])
a_new_query = MySQLQuery(financial_db_engine, db['dictionary_file_path'])
geo_df = pd.read_excel(r'geo_locations.xlsx')

stock_list, industry_list, d_12, d_23, d_3n, d_n1, d_n2, d_n3, d_nc = db_query.get_industry_info()

a_random_number = randint(0, len(stock_list))

header = html.Header('财务分析')

search_part = html.Nav(
    [
        html.Label('个股', className='nav_label'),
        dcc.Dropdown(id='stock_dropdown', className='nav_dropdown', options=stock_list, value=stock_list[a_random_number]),

        html.Label('一级行业', className='nav_label'),
        dcc.Dropdown(id='industry_dropdown', className='nav_dropdown', options=industry_list),

        html.Label('二级行业', className='nav_label'),
        dcc.RadioItems(id='industry_2_radio', className='nav_radio', inline=True),

        html.Label('三级行业', className='nav_label'),
        dcc.RadioItems(id='industry_3_radio', className='nav_radio', inline=True),

        html.Label('业内公司', className='nav_label'),
        dcc.Tabs(id='tabs_compare_by', className='nav_tabs', value='tab_industry',
                 children=[dcc.Tab(label='多选', className='nav_tabs', value='tab_industry', children=dcc.Checklist(id='stock_check', inline=True)),
                           dcc.Tab(label='单选', className='nav_tabs', value='tab_company', children=dcc.RadioItems(id='stock_radio', inline=True))])
    ],
    className='navigation'
)


result_part = html.Article(
    [
        html.Div(
            [
                html.H2(children='收入对比', id='title_income'),
                dcc.Loading(children=[dcc.Graph(id='income_bar', className='chart')], type='circle')
            ], className='chart_container'),

        html.Div(
            [
                html.H2(children='成本拆解', id='title_cost'),
                dcc.Loading(children=[dcc.Graph(id='cost_bar', className='chart')], type='circle')
            ], className='chart_container'),

        html.Div(
            [
                html.H2(children='运营效率', id='title_efficiency'),
                dcc.Graph(id='efficiency_bar', className='chart')
            ], className='chart_container'),

        html.Div(
            [
                html.H2(children='运营资产弹性', id='title_warren'),
                dcc.Graph(id='warren_bar', className='chart')
            ], className='chart_container'),

        html.Div(
            [
                html.H2(children='历史估值', id='title_valuation'),
                dcc.Graph(id='valuation_line', className='chart-tall')
            ], className='chart_container'),

        html.Div(
            [
                html.H2(children='产业地图', id='title_map'),
                dcc.Graph(id='map', className='map')
            ]),

        dash_table.DataTable(
            id='pe_table',
            style_table={'width': 500, 'margin': 60},
            style_cell={'textAlign': 'center', 'whiteSpace': 'pre-line'},
            tooltip_duration=None,
            columns=[
                {'id': '公司', 'name': '公司', 'presentation': 'markdown'},
                {'id': '价格', 'name': '价格'},
                {'id': 'PE', 'name': 'PE'},
                {'id': 'PE暗示的成长性', 'name': 'PE暗示的成长性'},
                {'id': '实际成长性（EPS）', 'name': '实际成长性（EPS）'}
            ]
        )
    ]
)


aside_part = html.Aside(
    [
        html.H3('快速直达'),
        html.Div(html.A(children='收入对比', href='#title_income'), className='aside_item'),
        html.Div(html.A(children='成本拆解', href='#title_cost'), className='aside_item'),
        html.Div(html.A(children='运营效率', href='#title_efficiency'), className='aside_item'),
        html.Div(html.A(children='资产弹性', href='#title_warren'), className='aside_item'),
        html.Div(html.A(children='历史估值', href='#title_valuation'), className='aside_item'),
        html.Div(html.A(children='产业地图', href='#title_map'), className='aside_item'),
        html.Div(html.A(children='成长性数据', href='#pe_table'), className='aside_item')
    ], className='aside'
)

main = html.Main([search_part, result_part, aside_part], style={'display': 'flex'})

app = dash.Dash(__name__)
app.layout = html.Div([header, main])


# 选择公司
@app.callback(Output('industry_dropdown', 'value'),
              Output('industry_2_radio', 'value'),
              Output('industry_3_radio', 'value'),
              Input('stock_dropdown', 'value'))
def stock_trigger(stock_name):
    industry_name = d_n1[stock_name]
    industry_2_value = d_n2[stock_name]
    industry_3_value = d_n3[stock_name]
    return industry_name, industry_2_value, industry_3_value


@app.callback(Output('industry_2_radio', 'options'), Input('industry_dropdown', 'value'))
def industry_trigger(industry_name):
    return d_12[industry_name]


@app.callback(Output('industry_3_radio', 'options'), Input('industry_2_radio', 'value'))
def industry_2_trigger(industry_2_name):
    return d_23[industry_2_name]


@app.callback(Output('stock_check', 'options'),
              Output('stock_check', 'value'),
              Output('stock_radio', 'options'),
              Output('stock_radio', 'value'),
              Input('industry_3_radio', 'value'),
              Input('stock_dropdown', 'value'))
def industry_3_trigger(industry_3_name, stock_name):
    companies_in_this_industry = db_query.sort_by_revenue(d_3n[industry_3_name])
    checked = companies_in_this_industry[:15]

    if stock_name in companies_in_this_industry and stock_name not in checked:
        checked.append(stock_name)

    if stock_name in companies_in_this_industry:
        selected_radio = stock_name
    else:
        selected_radio = companies_in_this_industry[0]

    return companies_in_this_industry, checked, companies_in_this_industry, selected_radio


# 画图
@app.callback(Output('income_bar', 'figure'),
              Output('cost_bar', 'figure'),
              Output('efficiency_bar', 'figure'),
              Output('warren_bar', 'figure'),
              Input('tabs_compare_by', 'value'),
              Input('industry_3_radio', 'value'),
              Input('stock_check', 'value'),
              Input('stock_radio', 'value'))
def update_charts(compared_by, industry_3_name, secname_list, company_name):

    if compared_by == 'tab_company':
        secname_list = [company_name]
        industry_3_name = company_name

    # 获取数据
    kpi_df = db_query.get_kpi(secname_list)
    income_df = db_query.get_income(secname_list)

    # income图
    income_fig = income_chart(kpi_df, industry_3_name)

    # cost图
    cost_fig = cost_chart(income_df, industry_3_name)

    # 运营效率图
    efficiency_fig = efficiency_chart(kpi_df, industry_3_name)

    # 巴菲特金手指图。这个图始终展示单一公司的数据
    warren_fig = warren_touch_chart(db_query.get_kpi([company_name]), company_name)

    return income_fig, cost_fig, efficiency_fig, warren_fig


# 历史估值图。因为画图方法比较独立，所以单独拎出来。
@app.callback(Output('valuation_line', 'figure'), Input('stock_radio', 'value'))
def valuation_chart(company_name):

    # 从数据库获取数据
    # income_df = db_query.get_income([company_name], annual_only=False, reindex_year_and_name=False).set_index('报告年度')
    income_df = a_new_query.get_income([company_name], reindex_year_and_name=False).set_index('报告年度')
    price_df = a_new_query.history_price([company_name]).set_index('交易日期')

    # 把计算出来的静态、TTM、预测EPS数据、股价、PE数据，加进df里
    combined_df = add_eps_pe(income_df, price_df)

    # 每季度采一个样应该够了。BQS表示 Business Quarter Start
    # tail(40)：最多40个季度(即10年)的数据。时间太久远意义不大而且图不好看了。
    # chart_df = combined_df.resample('BQS').nearest().tail(40)
    chart_df = combined_df.resample('A-MAY', convention='end').nearest().loc[:datetime.today()]

    # 画图。尝试过画在一起但大小差异很大，不好看。因此分成三个独立的图。
    fig = make_subplots(rows=3, cols=1, subplot_titles=['每股收益（TTM）', '股价', '市盈率（TTM）'], vertical_spacing=0.1)
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['eps_ttm'], mode='lines+markers+text', text=chart_df['eps_ttm'], textposition='top center'), row=1, col=1)
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['昨日收盘价'], mode='lines+markers+text', text=chart_df['昨日收盘价'].round(1), textposition='top center'), row=2, col=1)
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['pe_ttm'], mode='lines+markers+text', text=chart_df['pe_ttm'].round(), textposition='top center'), row=3, col=1)

    fig.update_layout(height=800, width=len(chart_df)*60, title_text=company_name, showlegend=False)

    return fig


# 更新表
@app.callback(Output('pe_table', 'data'),
              Output('pe_table', 'tooltip_data'),
              Input('tabs_compare_by', 'value'),
              Input('stock_check', 'value'),
              Input('stock_radio', 'value'))
def update_table(compared_by, secname_list, company_name):

    if compared_by == 'tab_company':
        secname_list = [company_name]

    # 获取数据
    price_df = db_query.history_price(secname_list)
    quarterly_income = db_query.get_income(secname_list, annual_only=False, reindex_year_and_name=False)

    table_data = []
    tooltip_data = []
    for secname in secname_list:

        report = r'[{}]({})'.format(secname, annual_report_link(d_nc[secname]))

        # 获取几个每股收益（EPS）数据
        single_company_income = quarterly_income.query('证券简称=="{}"'.format(secname)).set_index('报告年度')
        single_company_price = price_df.query('证券简称=="{}"'.format(secname)).set_index('交易日期')

        last_row = add_eps_pe(single_company_income, single_company_price).reset_index().iloc[-1]
        diluted_eps = single_company_income.combined_eps.squeeze()

        data_row = {'公司': report,

                    '价格': last_row['昨日收盘价'],

                    'PE': '静态：{}\nTTM：{}\n预测：{}'.format(last_row.pe_static, last_row.pe_ttm, last_row.pe_forcast),

                    'PE暗示的成长性':
                        '静态：{}\nTTM：{}\n预测：{}'.format(
                            suggested_growth(last_row.pe_static),
                            suggested_growth(last_row.pe_ttm),
                            suggested_growth(last_row.pe_forcast)
                        ),

                    '实际成长性（EPS）':
                        '{}\n{}\n{}'.format(
                            actual_growth(diluted_eps),
                            actual_growth(diluted_eps, 10),
                            actual_growth(diluted_eps, 5)
                        )

                    }

        tooltip_row = {
            'PE':
                {'value': '静态 EPS：{}\\\nTTM EPS：{}\\\n预测 EPS：{}'.format(last_row.eps_static, last_row.eps_ttm, last_row.eps_forcast),
                'type': 'markdown'},
            '价格':
                '日期：{}'.format(last_row['交易日期'].strftime('%Y-%m-%d'))}

        table_data.append(data_row)
        tooltip_data.append(tooltip_row)

    return table_data, tooltip_data


# 地图
@app.callback(Output('map', 'figure'), Input('industry_dropdown', 'value'))
def update_map(industry_1_name):

    # 获取数据
    location_df = db_query.get_location_and_profit(industry_1_name)

    # 附加GPS信息
    merged_df = location_df.merge(geo_df, left_on='所属城市', right_on='区域名称').dropna()

    return map_figure(merged_df, industry_1_name)


if __name__ == '__main__':
    app.run_server(debug=True)

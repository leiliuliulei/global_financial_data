from datetime import datetime
from sqlalchemy import create_engine
from random import randint
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
from db_functions import CnInfoAPI, MySQLQuery, translation_dictionary, income_chart, cost_chart, efficiency_chart, \
    warren_touch_chart, annual_report_link, static_ttm_forcast, actual_growth, suggested_growth

my_key, my_secret = 'fa4e980eaf7e4302811fb72336a648d0', '939563c8f0df4092b58823ae0d53ccb0'
financial_db_engine = create_engine('mysql+pymysql://root:LaZhu_007@localhost:3306/financial_data')

db_query = MySQLQuery(financial_db_engine, translation_dictionary())
# api_query = CnInfoAPI(my_key, my_secret, translation_dictionary())

stock_list, industry_list, d_12, d_23, d_3n, d_n1, d_n2, d_n3, d_nc = db_query.get_industry_info()


header = html.H2('财务分析', style={'textAlign': 'center'})


search_stock_left = html.Div(
    [
        html.Label('个股'),
        dcc.Dropdown(id='stock_dropdown', options=stock_list, value=stock_list[randint(0, 2000)]),

        html.Br(),
        html.Br(),
        html.Label('一级行业'), dcc.Dropdown(id='industry_dropdown', options=industry_list),
    ],
    style={'padding': 10, 'flex': 1})

search_stock_right = html.Div(
    [
        html.Label('二级行业'), dcc.RadioItems(id='industry_2_radio', inline=False),
        html.Br(), html.Br(), html.Br(), html.Label('三级行业'), dcc.RadioItems(id='industry_3_radio')
    ],
    style={'padding': 10, 'flex': 5})

search_stock = html.Div([search_stock_left, search_stock_right], style={'display': 'flex', 'flex-direction': 'row'})

choose_stock = html.Div(
    [
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        dcc.Tabs(id='tabs_compare_by', value='tab_industry',
                 children=[dcc.Tab(label='全行业', value='tab_industry', children=dcc.Checklist(id='stock_check')),
                           dcc.Tab(label='单一公司', value='tab_company', children=dcc.RadioItems(id='stock_radio'))]),
        html.Br(),
        html.Br()
    ]
)


charts = html.Div(
    [
        dcc.Loading(children=[dcc.Graph(id='income_bar')], type='circle'),
        dcc.Loading(children=[dcc.Graph(id='cost_bar')], type='circle'),
        dcc.Graph(id='efficiency_bar'),
        dcc.Graph(id='warren_bar'),
    ]
)


value_table = html.Div(
    [
        dash_table.DataTable(
            id='pe_table',
            style_cell={'textAlign': 'center', 'whiteSpace': 'pre-line'},
            tooltip_duration=None,
            columns=[{'id': '公司', 'name': '公司', 'presentation': 'markdown'},
                     {'id': '价格', 'name': '价格'},
                     {'id': 'PE', 'name': 'PE'},
                     {'id': 'PE暗示的成长性', 'name': 'PE暗示的成长性'},
                     {'id': '实际成长性（EPS）', 'name': '实际成长性（EPS）'}
                     ]
        )
    ],
    style={'width': 500}
)


external_stylesheets = ['https://cdn.jsdelivr.net/npm/water.css@2/out/water.css']
app = dash.Dash(__name__)
app.layout = html.Div([header, search_stock, choose_stock, charts, value_table])


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
    price_df = db_query.latest_price(secname_list)
    quarterly_income = db_query.get_income(secname_list, annual_only=False, reindex_year_and_name=False)

    table_data = []
    tooltip_data = []
    for secname in secname_list:

        report = r'[{}]({})'.format(secname, annual_report_link(d_nc[secname]))

        # 获取稀释每股收益（diluted EPS）数据
        # diluted_eps那列有时为空，这时就用basic_eps那列数据来填充。正好有一个combine_first函数执行这个逻辑
        single_company_df = quarterly_income.query('证券简称=="{}"'.format(secname)).set_index('报告年度')
        basic_eps = single_company_df['（一）基本每股收益']
        diluted_eps = single_company_df['（二）稀释每股收益'].combine_first(basic_eps)

        static_eps, ttm_eps, forcast_eps = static_ttm_forcast(diluted_eps, datetime.today())

        try:
            price_row = price_df.loc[secname]
        except KeyError:
            its_price = static_pe = ttm_pe = forcast_pe = '未获取价格'
            price_date = None
        else:
            its_price = price_row['昨收盘']
            price_date = price_row['交易日期'].strftime('%Y-%m-%d')
            static_pe, ttm_pe, forcast_pe = [round(its_price/v, 1) if v else None for v in [static_eps, ttm_eps, forcast_eps]]

        data_row = {'公司': report,

                    '价格': its_price,

                    'PE': '静态：{}\nTTM：{}\n预测：{}'.format(static_pe, ttm_pe, forcast_pe),

                    'PE暗示的成长性':
                        '静态：{}\nTTM：{}\n预测：{}'.format(
                            suggested_growth(static_pe),
                            suggested_growth(ttm_pe),
                            suggested_growth(forcast_pe)
                        ),

                    '实际成长性（EPS）':
                        '{}\n{}\n{}'.format(
                            actual_growth(diluted_eps),
                            actual_growth(diluted_eps, 10),
                            actual_growth(diluted_eps, 5)
                        )

                    }

        tooltip_row = {'PE': {'value': '静态 EPS：{}\\\nTTM EPS：{}\\\n预测 EPS：{}'.format(static_eps, ttm_eps, forcast_eps),
                              'type': 'markdown'},

                       '价格': '日期：{}'.format(price_date)}

        table_data.append(data_row)
        tooltip_data.append(tooltip_row)

    return table_data, tooltip_data


if __name__ == '__main__':
    app.run_server(debug=True)

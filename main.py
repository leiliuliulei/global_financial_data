import dash
from random import choice
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
from analyser import DataA, DataH

segment_a = DataA(None).segment_info()
segment_h = DataH(None).segment_info()

stock_list_a = segment_a['stocks']
stock_list_h = segment_h['stocks']
stock_list = [f'{name} A' for name in stock_list_a] + [f'{name} H' for name in stock_list_h]

header = html.Header('财务分析')

search_part = html.Nav(
    [
        html.Label('个股', className='nav_label'),
        dcc.Dropdown(id='stock_dropdown', className='nav_dropdown', options=stock_list, value=choice(stock_list)),

        html.Label('一级行业', className='nav_label'),
        dcc.Dropdown(id='industry_dropdown', className='nav_dropdown'),

        html.Label('二级行业', className='nav_label'),
        dcc.RadioItems(id='industry_2_radio', className='nav_radio', inline=True),

        html.Label('三级行业', className='nav_label'),
        dcc.RadioItems(id='industry_3_radio', className='nav_radio', inline=True),

        html.Label('业内公司', className='nav_label'),
        dcc.Tabs(id='tabs_compare_by', className='nav_tabs', value='tab_industry',
                 children=[
                     dcc.Tab(label='多选', className='nav_tabs', value='tab_industry', children=dcc.Checklist(id='stock_check', inline=True)),
                     dcc.Tab(label='单选', className='nav_tabs', value='tab_company', children=dcc.RadioItems(id='stock_radio', inline=True))
                 ]),

        dcc.Store(id='market')
    ],
    className='navigation')


result_part = html.Article(
    [
        html.Div(
            [
                html.H2(children='收入对比', id='title_income'),
                dcc.Loading(children=[dcc.Graph(id='income_bar', className='chart')], type='circle')
            ],
            className='chart_container'),

        html.Div(
            [
                html.H2(children='成本拆解', id='title_cost'),
                dcc.Loading(children=[dcc.Graph(id='cost_bar', className='chart')], type='circle')
            ],
            className='chart_container',
        ),

        html.Div([html.H2(children='运营效率', id='title_efficiency'), dcc.Graph(id='efficiency_bar', className='chart')],
                 className='chart_container'),

        html.Div([html.H2(children='运营资产弹性', id='title_warren'), dcc.Graph(id='warren_bar', className='chart')],
                 className='chart_container'),

        html.Div([html.H2(children='历史估值', id='title_valuation'), dcc.Graph(id='valuation_line', className='chart-tall')],
                 className='chart_container'),

        dash_table.DataTable(id='pe_table', style_table={'width': 500, 'margin': 60},
                             style_cell={'textAlign': 'center', 'whiteSpace': 'pre-line'}, tooltip_duration=None,
                             columns=[
                                 {'id': '公司', 'name': '公司', 'presentation': 'markdown'},
                                 {'id': '价格', 'name': '价格'},
                                 {'id': 'PE', 'name': 'PE'},
                                 {'id': 'PE暗示的成长性', 'name': 'PE暗示的成长性'},
                                 {'id': '实际EPS成长性', 'name': '实际EPS成长性'}]),

        html.Div([html.H2(children='产业地图', id='title_map'), dcc.Graph(id='map', className='map')])
    ]
)


aside_part = html.Aside(
    [
        html.H3('快速直达'),
        html.Div(html.A(children='收入对比', href='#title_income'), className='aside_item'),
        html.Div(html.A(children='成本拆解', href='#title_cost'), className='aside_item'),
        html.Div(html.A(children='运营效率', href='#title_efficiency'), className='aside_item'),
        html.Div(html.A(children='资产弹性', href='#title_warren'), className="aside_item"),
        html.Div(html.A(children='历史估值', href='#title_valuation'), className='aside_item'),
        html.Div(html.A(children='成长性', href='#pe_table'), className='aside_item'),
        html.Div(html.A(children='产业地图', href='#title_map'), className='aside_item')
    ],
    className='aside',
)

main = html.Main([search_part, result_part, aside_part], style={'display': 'flex'})

app = dash.Dash(__name__)
app.layout = html.Div([header, main])


# 通过公司触发对行业的查询
@app.callback(Output('market', 'data'),
              Output('industry_dropdown', 'options'),
              Output('industry_dropdown', 'value'),
              Output('industry_2_radio', 'value'),
              Output('industry_3_radio', 'value'),
              Input('stock_dropdown', 'value'))
def stock_trigger(stock_name):

    stock_name = stock_name[:-2]            # 去除股票名称最后面跟着的 A 或者 H 字样

    if stock_name in stock_list_a:
        market = 'A股'
        industry_options = segment_a['industry_1']
        industry_name = segment_a['d_n1'][stock_name]
        industry_2_value = segment_a['d_n2'][stock_name]
        industry_3_value = segment_a['d_n3'][stock_name]

    elif stock_name in stock_list_h:
        market = '港股'
        industry_options = segment_h['industry_1']
        industry_name = segment_h['d_n1'][stock_name]
        industry_2_value = segment_h['d_n2'][stock_name]
        industry_3_value = None

    else:
        market = None
        industry_options = None
        industry_name = None
        industry_2_value = None
        industry_3_value = None

    return market, industry_options, industry_name, industry_2_value, industry_3_value


@app.callback(Output('industry_2_radio', 'options'),
              Input('industry_dropdown', 'value'),
              State('market', 'data'))
def industry_trigger(industry_name, market):
    if market == 'A股':
        return segment_a['d_12'][industry_name]
    else:
        return segment_h['d_12'][industry_name]


@app.callback(Output('industry_3_radio', 'options'),
              Input('industry_2_radio', 'value'),
              State('market', 'data'))
def industry_2_trigger(industry_2_name, market):
    if market == 'A股':
        return segment_a['d_23'][industry_2_name]
    else:
        return []


@app.callback(Output('stock_check', 'options'),
              Output('stock_check', 'value'),
              Output('stock_radio', 'options'),
              Output('stock_radio', 'value'),
              Input('industry_2_radio', 'value'),
              Input('industry_3_radio', 'value'),
              State('stock_dropdown', 'value'),
              State('market', 'data'))
def industry_3_trigger(industry_2_name, industry_3_name, stock_name, market):

    if market == 'A股':
        all_companies = segment_a['d_3n'][industry_3_name]
        sorted_companies = DataA(all_companies).sort()
    else:
        all_companies = segment_h['d_3n'][industry_2_name]
        sorted_companies = DataH(all_companies).sort()

    to_be_checked = sorted_companies[:15]

    stock_name = stock_name[:-2]            # 去除股票名称最后面跟着的 A 或者 H 字样
    if stock_name in sorted_companies and stock_name not in to_be_checked:
        to_be_checked.append(stock_name)

    # 逻辑：如果搜索框里的公司属于当前行业，则选择该公司；如果不属于，说明用户直接换了一个行业，则选择行业第一的公司
    to_be_selected = stock_name if stock_name in sorted_companies else sorted_companies[0]

    return sorted_companies, to_be_checked, sorted_companies, to_be_selected


# income做为第一个图，先单独画出来。用户感受更好。
@app.callback(
    Output('income_bar', 'figure'),
    Input('tabs_compare_by', 'value'),
    Input('stock_check', 'value'),
    Input('stock_radio', 'value'),
    State('industry_2_radio', 'value'),
    State('industry_3_radio', 'value'),
    State('market', 'data')
)
def income_chart(compared_by, company_list, company, industry_2_name, industry_3_name, market):

    target = company if compared_by == 'tab_company' else company_list

    if market == 'A股':
        return DataA(target).income_fig(industry_3_name)
    else:
        return DataH(target).income_fig(industry_2_name)


# 画完income图之后再画其他图。因此，这里的input用到了上面的图的output
@app.callback(Output('cost_bar', 'figure'),
              Output('efficiency_bar', 'figure'),
              Output('warren_bar', 'figure'),
              Input('income_bar', 'figure'),
              Input('tabs_compare_by', 'value'),
              Input('stock_check', 'value'),
              Input('stock_radio', 'value'),
              State('industry_2_radio', 'value'),
              State('industry_3_radio', 'value'),
              State('market', 'data'))
def other_basic_charts(income_figure, compared_by, company_list, company, industry_2_name, industry_3_name, market):

    target = company if compared_by == 'tab_company' else company_list

    if market == 'A股':
        f2 = DataA(target).cost_fig(industry_3_name)
        f3 = DataA(target).efficiency_fig(industry_3_name)
        f4 = DataA(company).warren_fig()
    else:
        f2 = DataH(target).cost_fig(industry_2_name)
        f3 = DataH(target).efficiency_fig(industry_2_name)
        f4 = DataH(company).warren_fig()

    return f2, f3, f4


# 成长性分析表
@app.callback(Output('valuation_line', 'figure'),
              Output('pe_table', 'data'),
              Output('pe_table', 'tooltip_data'),
              Input('income_bar', 'figure'),
              Input('stock_check', 'value'),
              Input('stock_radio', 'value'),
              State('market', 'data'))
def growth_and_valuation(income_figure, secname_list, company_name, market):

    if market == 'A股':
        f = DataA(company_name).valuation_fig()                     # 历史估值图
        data, tooltip = DataA(secname_list).growth_table()          # 成长性表格
    else:
        f = DataH(company_name).valuation_fig()
        data, tooltip = DataH(secname_list).growth_table()

    return f, data, tooltip


# 地图
@app.callback(Output('map', 'figure'),
              Input('income_bar', 'figure'),
              Input('industry_2_radio', 'value'),
              State('market', 'data'))
def industrial_map(income_figure, industry_2_name, market):

    if market == 'A股':
        return DataA(secname=None).industry_map(industry_2_name)
    else:
        return None


if __name__ == "__main__":
    app.run_server(debug=True)

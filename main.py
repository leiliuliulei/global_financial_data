import dash
from random import choice
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
from analyser import WebFig
from core.database import Segment

seg = Segment()
stock_list = [f'{name} A' for name in seg.stocks(market='A')] + [f'{name} H' for name in seg.stocks(market='H')]

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

        dash_table.DataTable(id='pe_table', style_table={'width': 1000, 'margin': 60},
                             style_cell={'textAlign': 'center', 'whiteSpace': 'pre-line'}, tooltip_duration=None,
                             columns=[
                                 {'id': '公司', 'name': '公司', 'presentation': 'markdown'},
                                 {'id': '上市年限', 'name': '上市年限'},
                                 {'id': '价格', 'name': '价格'},
                                 {'id': '毛利率', 'name': '毛利率'},
                                 {'id': '净利率', 'name': '净利率'},
                                 {'id': 'ROE', 'name': 'ROE'},
                                 {'id': '资产负债比', 'name': '资产负债比'},
                                 {'id': 'PE', 'name': 'PE'},
                                 {'id': 'EPS成长性', 'name': 'EPS成长性'},
                                 {'id': '估值gap', 'name': '估值gap'},
                                 {'id': '总收益', 'name': '总收益'}]),

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

app = dash.Dash(__name__, prevent_initial_callbacks='initial_duplicate')
app.layout = html.Div([header, main])


# 通过公司触发对行业的查询
@app.callback(Output('market', 'data'),
              Output('industry_dropdown', 'options'),
              Output('industry_dropdown', 'value'),
              Output('industry_2_radio', 'value'),
              Output('industry_3_radio', 'value'),
              Input('stock_dropdown', 'value'))
def stock_trigger(stock_name):

    # 把从界面获得的股票字符串拆分成纯名字和market。然后获得对应的行业信息。
    name, market = str(stock_name).split()
    seccode = seg.code(name, market)
    industry_1_options = seg.industry_list(market=market)
    industry_1_value, industry_2_value, industry_3_value = seg.belong(seccode)

    return market, industry_1_options, industry_1_value, industry_2_value, industry_3_value


@app.callback(Output('industry_2_radio', 'options'),
              Input('industry_dropdown', 'value'),
              State('market', 'data'))
def industry_trigger(industry_1_value, market):
    industry_2_options = seg.d12(industry_1_value, market)
    return industry_2_options


@app.callback(Output('industry_3_radio', 'options'),
              Input('industry_2_radio', 'value'),
              State('market', 'data'))
def industry_2_trigger(industry_2_value, market):
    industry_3_options = seg.d23(industry_2_value, market)
    return industry_3_options


@app.callback(Output('stock_check', 'options'),
              Output('stock_check', 'value'),
              Output('stock_radio', 'options'),
              Output('stock_radio', 'value'),
              Input('industry_2_radio', 'value'),
              Input('industry_3_radio', 'value'),
              State('stock_dropdown', 'value'),
              State('market', 'data'))
def industry_3_trigger(industry_2_value, industry_3_value, stock_name, market):

    companies = seg.d3c(industry_2_value, industry_3_value, market)
    stock_code = seg.code(stock_name[:-2], market)

    # 复选逻辑：默认选取行业（收入）前15名。如果感兴趣的公司不在前15名，就把它加进去
    checked = companies[:15]
    if stock_code in companies and stock_code not in checked:
        checked.append(stock_code)

    # 单选逻辑：如果搜索框里的公司属于当前行业，则选择该公司；如果不属于，说明用户直接换了一个行业，则选择行业第一的公司
    single_selected = stock_code if stock_code in companies else companies[0]

    # 针对Dash的checkList对象进行优化：
    # label是界面上显示的文本，这里显示证券名称，方便阅读；value是checkList的实际值，选择seccode以方便程序后续处理。
    companies_options = [{'label': seg.name(code), 'value': code} for code in companies]

    return companies_options, checked, companies_options, single_selected


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

    if compared_by == 'tab_company':
        return WebFig(seccodes=company, market=market).income_fig(title=seg.name(company))
    else:
        title = industry_3_name if industry_3_name else industry_2_name
        return WebFig(seccodes=company_list, market=market).income_fig(title=title)


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

    if compared_by == 'tab_company':
        fig_engine = WebFig(seccodes=company, market=market)
        f2 = fig_engine.cost_fig(title=seg.name(company))
        f3 = fig_engine.efficiency_fig(title=seg.name(company))
        f4 = fig_engine.warren_fig(title=seg.name(company))
    else:
        fig_engine = WebFig(seccodes=company_list, market=market)
        title = industry_3_name if industry_3_name else industry_2_name
        f2 = fig_engine.cost_fig(title=title)
        f3 = fig_engine.efficiency_fig(title=title)

        f4 = WebFig(seccodes=company, market=market).warren_fig(title=seg.name(company))

    return f2, f3, f4


# 成长性分析表
@app.callback(Output('valuation_line', 'figure'),
              Output('pe_table', 'data'),
              Output('pe_table', 'tooltip_data'),
              Input('income_bar', 'figure'),
              Input('stock_check', 'value'),
              Input('stock_radio', 'value'),
              State('market', 'data'))
def growth_and_valuation(income_figure, company_list, company, market):

    fig = WebFig(seccodes=company, market=market).valuation_fig(title=seg.name(company))
    data, tooltip = WebFig(seccodes=company_list, market=market).gui_table()

    return fig, data, tooltip


# 地图
@app.callback(Output('map', 'figure'),
              Input('income_bar', 'figure'),
              Input('industry_2_radio', 'value'),
              State('market', 'data'))
def industrial_map(income_figure, industry_2_name, market):
    return WebFig(seccodes=None, market=market).industry_map(industry_2_name)


if __name__ == "__main__":
    app.run_server(debug=True)

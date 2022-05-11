from sqlalchemy import create_engine
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from db_functions import industry_mapping, rank_by_revenue, get_financial_data, translation_dictionary, \
    income_chart, cost_chart, efficiency_chart, warren_touch_chart, balance_sheet_chart

engine = create_engine('mysql+pymysql://root:LaZhu_007@localhost:3306/financial_data')
stock_list, industry_list, d_12, d_23, d_3n, d_n1, d_n2, d_n3, d_nc = industry_mapping(engine)
name_dict = translation_dictionary()

header = html.H2('财务分析', style={'textAlign': 'center'})

choose_stock_left = html.Div(
    [
        html.Label('个股'), dcc.Dropdown(id='stock_dropdown', options=stock_list, value=stock_list[100]),
        html.Br(), html.Br(), html.Label('一级行业'), dcc.Dropdown(id='industry_dropdown', options=industry_list),
        html.Br(), html.Br(), html.Label('二级行业'), dcc.RadioItems(id='industry_2_radio', inline=False)
    ],
    style={'padding': 10, 'flex': 1})

choose_stock_right = html.Div(
    [
        html.Label('三级行业'), dcc.RadioItems(id='industry_3_radio'),
        html.Br(), html.Br(), html.Label('同行'), dcc.Checklist(id='stock_check')
    ],
    style={'padding': 10, 'flex': 1})

choose_stock = html.Div([choose_stock_left, choose_stock_right], style={'display': 'flex', 'flex-direction': 'row'})

# df_memory = html.Div([dcc.Store(id='df_memory_2301'), dcc.Store(id='df_memory_2303')])

company_switch = html.Div(
    [
        html.Br(),
        html.Br(),
        dcc.RadioItems(id='industry_or_company', options={'industry': '本行业', 'company': '仅此公司'}, value='industry')
    ])
# charts = html.Div([dcc.Graph(id='income_bar'), dcc.Graph(id='cost_bar'), dcc.Graph(id='efficiency_bar'),
#                    dcc.Graph(id='warren_bar')])
charts = html.Div(
    [
        dcc.Graph(id='income_bar'),
        dcc.Graph(id='cost_bar'),
        dcc.Graph(id='efficiency_bar'),
        dcc.Graph(id='warren_bar'),
        dcc.Graph(id='balance_bar')
    ]
)

app = dash.Dash(__name__)
app.layout = html.Div([header, choose_stock, company_switch, charts])


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
              Input('industry_3_radio', 'value'),
              Input('stock_dropdown', 'value'))
def industry_3_trigger(industry_3_name, stock_name):

    companies_in_this_industry = rank_by_revenue(d_3n[industry_3_name], engine)
    selected = companies_in_this_industry[:15]

    if stock_name in companies_in_this_industry and stock_name not in selected:
        selected.append(stock_name)

    return companies_in_this_industry, selected


# 画图
@app.callback(Output('income_bar', 'figure'),
              Output('cost_bar', 'figure'),
              Output('efficiency_bar', 'figure'),
              Output('warren_bar', 'figure'),
              Output('balance_bar', 'figure'),
              Input('stock_check', 'value'),
              Input('industry_or_company', 'value'),
              Input('stock_dropdown', 'value'))
def update_charts(secname_list, industry_or_company, company_name):

    df_2300_industry, df_2301_industry, df_2303_industry = get_financial_data(secname_list, name_dict, engine)
    df_2300_company, df_2301_company, df_2303_company = get_financial_data([company_name], name_dict, engine)

    if industry_or_company == 'industry':

        income_fig = income_chart(df_2303_industry)
        cost_fig = cost_chart(df_2301_industry)
        efficiency_fig = efficiency_chart(df_2303_industry)
        balance_fig = balance_sheet_chart(df_2300_industry)

    else:
        income_fig = income_chart(df_2303_company)
        cost_fig = cost_chart(df_2301_company)
        efficiency_fig = efficiency_chart(df_2303_company)
        balance_fig = balance_sheet_chart(df_2300_company)

    warren_fig = warren_touch_chart(df_2303_company, company_name)

    return income_fig, cost_fig, efficiency_fig, warren_fig, balance_fig


# 选择公司
# @app.callback(Output('industry_dropdown', 'value'), Input('stock_dropdown', 'value'))
# def update_industry_dropdown(stock_name):
#     return d_n1[stock_name]
#
#
# @app.callback(Output('industry_2_radio', 'options'), Output('industry_2_radio', 'value'),
#               Input('industry_dropdown', 'value'), Input('stock_dropdown', 'value'))
# def update_industry_2_radio(industry_name, stock_name):
#     return d_12[industry_name], d_n2[stock_name]
#
#
# @app.callback(Output('industry_3_radio', 'options'), Output('industry_3_radio', 'value'),
#               Input('industry_2_radio', 'value'), Input('stock_dropdown', 'value'))
# def update_industry_3_radio(industry_2_name, stock_name):
#     return d_23[industry_2_name], d_n3[stock_name]
#
#
# @app.callback(Output('stock_check', 'options'), Output('stock_check', 'value'),
#               Input('industry_3_radio', 'value'), Input('stock_dropdown', 'value'))
# def update_stock_checklist(industry_3_name, stock_name):
#     companies_in_this_industry = rank_by_revenue(d_3n[industry_3_name], engine)
#     selected = companies_in_this_industry[:10]
#     if stock_name not in selected:
#         selected.append(stock_name)
#     return companies_in_this_industry, selected


# 数据存储
# @app.callback(Output('df_memory_2301', 'data'), Output('df_memory_2303', 'data'), Input('stock_check', 'value'))
# def query_financial_data(secname_list):
#     df_2301, df_2303 = get_financial_data(secname_list, engine)
#
#     data_2301 = df_2301.rename(columns=name_dict['stock2301']).to_dict('records')
#     data_2303 = df_2303.rename(columns=name_dict['stock2303']).to_dict('records')
#
#     return data_2301, data_2303


# # cost图
# @app.callback(Output('cost_bar', 'figure'),
#               Input('df_memory_2301', 'data'),
#               Input('industry_or_company', 'value'),
#               Input('stock_dropdown', 'value'))
# def update_cost_chart(df_memory_data, industry_or_company, company_name):
#     return cost_chart(df_memory_data, industry_or_company, company_name)


# 巴菲特金手指（运营资金和收入、利润关系）
# @app.callback(Output('warren_bar', 'figure'),
#               Input('df_memory_2303', 'data'),
#               Input('stock_dropdown', 'value'))
# def update_warren_chart(df_memory_data, company_name):
#     return warren_touch_chart(df_memory_data, company_name)


if __name__ == '__main__':
    app.run_server(debug=True)

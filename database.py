import toml
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine


class FinancialDatabase(object):

    def __init__(self):

        # 加载配置
        config = toml.load('configuration.toml')

        # 创建数据库 engine
        db = config['database']
        conn_string = f"mysql+pymysql://{db['account']}:{db['password']}@{db['address']}:{db['port']}/{db['name']}"
        self._engine = create_engine(conn_string)

        # 数据库名称字典
        dict_setting = config['dictionary']
        dfs = pd.read_excel(dict_setting['dictionary_file_path'], sheet_name=None)
        del dfs['说明']

        self._db2real_dicts = {sheet: df.set_index('数据库名称')['中文名称'].squeeze().to_dict() for (sheet, df) in dfs.items()}
        self._real2db_dicts = {sheet: df.set_index('中文名称')['数据库名称'].squeeze().to_dict() for (sheet, df) in dfs.items()}

        self._table_specific_rule = {
            'stock2100': 'SECCODE REGEXP "^[0-9]" and CHANGE_CODE in (1, 3) and F034V is not null',
            'stock2301': 'F002V = "071001" and CHANGE_CODE in (1,3)',
            'stock2302': 'F002V = "071001" and CHANGE_CODE in (1,3)',
            'stock2303': 'F070V = "071001" and CHANGE_CODE in (1,3) and F001V = "033003"',
            'stock2401': '',
            'stock2402': '',
            'hk4001': '',
            'hk4023': '',
            'hk4024': '',
            'hk4025': '',
            'hk4026': ''}

    def query(self, query_string, table_name):
        df = pd.read_sql_query(query_string, self._engine, parse_dates=['ENDDATE', 'TRADEDATE', 'DECLAREDATE'])
        df = df.rename(columns=self._db2real_dicts[table_name])
        return df

    def query_secname(self, table, column_list, secname, annual=True):

        q_string = self._dynamic_query(table=table,
                                       column_list=column_list,
                                       secname=secname,
                                       annual=annual,
                                       last_year_only=False,
                                       sort_time=True,
                                       sort_income=False,
                                       sort_secname=True)

        return self.query(q_string, table)

    def sort_income(self, table, column_list, secname):

        q_string = self._dynamic_query(table=table,
                                       column_list=column_list,
                                       secname=secname,
                                       annual=False,
                                       last_year_only=True,
                                       sort_time=False,
                                       sort_income=True,
                                       sort_secname=False)

        return self.query(q_string, table)

    def query_without_sort(self, table, column_list, secname):

        q_string = self._dynamic_query(table=table,
                                       column_list=column_list,
                                       secname=secname,
                                       annual=False,
                                       last_year_only=False,
                                       sort_time=False,
                                       sort_income=False,
                                       sort_secname=False)

        return self.query(q_string, table)

    def query_location(self, industry_2):

        query_str = f'''
        
        SELECT s21.SECNAME, s21.SECCODE, s21.F028V, s21.F038V, s23.F006N 

            FROM stock2100 s21 INNER JOIN stock2301 s23 ON s21.SECCODE = s23.SECCODE 

            WHERE 
                s21.F036V = "{industry_2}" and 
                s21.CHANGE_CODE <> 2 and 
                s23.F001D = "{newest_fiscal_year()}-12-31" and 
                s23.F002V = "071001" and 
                s23.CHANGE_CODE <> 2;'''

        return self.query(query_str, 'stock2100')

    def _get_db_name(self, table_name, name):

        table_dict = self._real2db_dicts[table_name]

        if isinstance(name, list):
            return [table_dict[item] for item in name if item in table_dict]
        elif isinstance(name, str):
            return table_dict[name] if name in table_dict else None

    def _dynamic_query(self, table, column_list, secname, annual, last_year_only, sort_time, sort_income, sort_secname):

        # 这部分处理 select from
        column_str = ', '.join(self._get_db_name(table, column_list))
        select_str = f'select {column_str} from {table}'

        # 这部分处理 where
        if secname:
            name_str = ', '.join(double_quote(secname))
            name_rule = f'SECNAME in ({name_str})'
        else:
            name_rule = None

        if self._get_db_name(table, '截止日期'):
            time_name = self._get_db_name(table, '截止日期')
        else:
            time_name = self._get_db_name(table, '交易日期')

        annual_rule = f'{time_name} REGEXP "12-31"' if annual else None
        latest_rule = f'{time_name} = "{newest_fiscal_year()}-12-31"' if last_year_only else None
        other_rule = self._table_specific_rule[table]

        rule_list = [item for item in [name_rule, annual_rule, latest_rule, other_rule] if item]

        if rule_list:
            rule_str = ' and '.join(rule_list)
            where_str = f'where {rule_str}'
        else:
            where_str = ''

        # 这部分处理 sort
        time_str = f'{time_name} desc' if sort_time else None

        if sort_income:
            income_candidate = self._get_db_name(table, ['营业总收入', '营业额'])
            income_str = f'{income_candidate[0]} desc'
        else:
            income_str = None

        if sort_secname:
            # case_list形成 ['when "招商银行" then 1', 'when "格力电器" then 2']
            when_list = [f'when {item} then {i}' for i, item in enumerate(double_quote(secname), start=1)]
            when_str = ' '.join(when_list)
            case_str = f'case SECNAME {when_str} end'
        else:
            case_str = None

        sort_list = [item for item in [time_str, income_str, case_str] if item]

        if sort_list:
            sort_str = f'order by {", ".join(sort_list)}'
        else:
            sort_str = ''

        # 最后整合三个部分
        sql_string = f'{select_str} {where_str} {sort_str}'.strip() + ';'

        return sql_string


def double_quote(list_or_string):

    # 若输入的是list则直接用。若不是，外面套一个[]变成list
    work_list = list_or_string if isinstance(list_or_string, list) else [list_or_string]

    # 给work_list里的element加上双引号。例如从 ['招商银行', '格力电器']变成了['"招商银行"', '"格力电器"']
    double_quote_list = [f'"{item}"' for item in work_list]

    return double_quote_list


def newest_fiscal_year():
    # 4月以后，去年年报都出来了，因此newest year是去年；反之，newest year是前年
    current_year, current_month = datetime.today().year, datetime.today().month
    newest_year = current_year - 1 if current_month > 4 else current_year - 2

    return newest_year



from database import Segment, Data, SingleProcessor
from analyser import Holdings


my_holdings = {'600519': 1100,
               '002597': 24000,
               '600036': 60000,
               '000651': 30100,
               '02020': 14600,
               '09618': 180,
               '01181': 646000,
               '00700': 5200,
               '00939': 58000}

a = Holdings(my_holdings)
a.holding_info(hkd_rate=0.9397).to_excel(r'C:\Users\Liu_Lei\Desktop\持仓_2024年12月25日.xlsx')

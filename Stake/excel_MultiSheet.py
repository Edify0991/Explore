import numpy as np
import openpyxl
import pandas as pd
# 打开excel文件
#wb = openpyxl.load_workbook('负债资产表.xlsx')
df = pd.read_excel("负债资产表.xlsx", engine="openpyxl", sheet_name=None)
writer = pd.ExcelWriter('esg.xlsx')
n = 1
# 遍历所有sheet
for i in df.keys():
    print(n)
    n += 1
    sheet_i = pd.read_excel("负债资产表.xlsx", engine = "openpyxl", sheet_name = i)
    row_name = sheet_i.columns[1 : ]
    total_liabilities = sheet_i.loc[122: 122, "20111231" : "20211231"].values
    if len(total_liabilities) == 0:
        break
    total_assets = sheet_i.loc[67: 67, "20111231" : "20211231"].values
    solvency = np.divide(total_liabilities, total_assets)
    company_size = np.log(total_assets)
    Data = np.concatenate((solvency, company_size), axis=-1)
    Data = Data.reshape(2, int(Data.shape[1] / 2))
    pd_data = pd.DataFrame(columns = row_name, index = ['solvency', 'company_size'], data = Data)
    pd_data.to_excel(writer, sheet_name = i)
    print(pd_data)
writer.close()
# 遍历每个sheet
#for sheet in sheets:
    # 获取当前sheet
#    ws = wb[sheet]
    # 定义一个变量，用来存储当前sheet中的数据
#    sheet_data = 0
    # 遍历每一行
#    for row in ws.rows:
        # 遍历每一列
#        for cell in row:
            # 将每个单元格的值相加


    # 将每个sheet的数据存储到字典中

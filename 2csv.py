import os
import pandas as pd

def generate_label_csv(data_path, output_csv_path):
    # 列出数据目录中的所有文件
    files = os.listdir(data_path)
    
    # 创建一个列表，用于存储标签和文件名
    data = []

    # 遍历文件，提取标签并存储到列表中
    for file in files:
        if 'AFIB' in file:
            label = 1
        else:
            label = 0
        
        data.append([label, file])
    
    # 创建一个 DataFrame
    df = pd.DataFrame(data, columns=['label', 'Filename'])
    
    # 保存到 CSV 文件
    df.to_csv(output_csv_path, index=False)

# 示例调用
data_path = './data/4/'  # 数据目录路径
output_csv_path = './data_indices/final_test_indice.csv'  # 输出 CSV 文件路径

generate_label_csv(data_path, output_csv_path)

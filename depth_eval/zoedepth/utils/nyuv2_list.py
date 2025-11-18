import os
import re

# 定义文件路径
test_file = './train_test_inputs/nyudepthv2_depthbin_train_files_with_gt.txt'
nyutest_folder = './data/nyu/NYUtrain'

# 定义正则表达式匹配文件名中的序号部分
pattern = re.compile(r'NYU(\d{4})_0000_color')

# 存储匹配的文件名
matched_files = []

# 遍历NYUtest文件夹中的所有文件
for file_name in os.listdir(nyutest_folder):
    if file_name.endswith('_color.jpg'):
        # 去掉.jpg后缀以便匹配.png文件
        base_name = os.path.splitext(file_name)[0]
        png_file = base_name.replace('_color', '') + '.png'

        # 检查jpg和png文件是否都存在
        if os.path.exists(os.path.join(nyutest_folder, file_name)) and os.path.exists(
                os.path.join(nyutest_folder, png_file)):
            # 提取序号部分
            match = pattern.search(file_name)
            if match:
                sequence_number = match.group(1)
                matched_files.append((sequence_number, file_name, png_file))

# 按序号排序
matched_files.sort(key=lambda x: int(x[0]))

# 将结果写入文件
with open(test_file, 'w') as out_f:
    for _, jpg_file, png_file in matched_files:
        out_f.write(f'{jpg_file} {png_file} 518.8579\n')

print(f'输出已写入 {test_file}')

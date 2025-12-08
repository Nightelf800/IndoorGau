import os
import json
import base64
import glob
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from volcenginesdkarkruntime import Ark


# 设置数据根目录
nyuv2_data_root = './data/NYUv2'

# 豆包API配置
API_MODEL = 'doubao-1.5-vision-lite-250315'
API_KEY = 'b32d69c3-1dd7-450a-a4cf-2672a7673001'  # 请替换为您自己的豆包API密钥
API_URL = 'hhttps://ark.cn-beijing.volces.com/api/v3/chat/completions'

# 生成caption的函数
def generate_caption(image_path):
    """
    使用豆包API生成图像的caption
    """
    try:

        # 读取图像
        with open(image_path, 'rb') as f:
            image_data = f.read()

        PROMPT = """
        You are an expert in image content recognition. Please analyze this indoor image and generate a detailed caption that includes:
        1. The objects present in the image and their attribute features
        2. The spatial locations of the objects
        3. The geometric relationships between the objects
        Provide a clear and concise description in a single paragraph.
        """

        client = Ark(api_key=API_KEY)

        completion = client.chat.completions.create(
            model=API_MODEL,
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {
                            'type': 'text',
                            'text': PROMPT
                        },
                        {
                            'type': 'image_url',
                            'image_url': {
                                'url': f'data:image/jpeg;base64,{base64.b64encode(image_data).decode()}'
                            }
                        }
                    ]
                }
            ]
        )

        # 获取响应
        caption = completion.choices[0].message.content
        completion_tokens = completion.usage['completion_tokens']
        prompt_tokens = completion.usage['prompt_tokens']
        
        return caption, completion_tokens, prompt_tokens
    except Exception as e:
        print(f"生成caption时出错 {image_path}: {e}")
        return None, 0, 0

# 处理数据集的函数
def process_dataset(data_split):
    """
    处理指定的数据分割（train或test）
    """
    # 设置输入和输出路径
    input_dir = os.path.join(nyuv2_data_root, 'depthbin', f'NYU{data_split}')
    output_dir = os.path.join(nyuv2_data_root, 'text', f'NYU{data_split}')
    output_file = os.path.join(output_dir, f'caption.json')
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有_color.jpg或_color.png图像
    image_files = glob.glob(os.path.join(input_dir, '*_color.jpg')) + glob.glob(os.path.join(input_dir, '*_color.png'))
    image_files = sorted(image_files)

    print(f'images.len: {len(image_files)}')
    
    # 生成caption并保存结果
    captions = {}
    for image_file in tqdm(image_files, desc=f'生成{data_split}数据集 caption'):
        # 获取basename（不含扩展名）
        basename = os.path.basename(image_file)
        key = basename.split('.')[0].replace('_color', '')
        
        # 生成caption
        caption, completion_tokens, prompt_tokens = generate_caption(image_file)
        if caption:
            captions[key] = {
                'caption': caption,
                'completion_tokens': completion_tokens,
                'prompt_tokens': prompt_tokens
            }
        else:
            captions[key] = {
                'caption': '',
                'completion_tokens': 0,
                'prompt_tokens': 0
            }
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(captions, f, ensure_ascii=False, indent=2)
    
    print(f"{data_split}数据集处理完成，结果保存在 {output_file}")

if __name__ == '__main__':
    # 处理训练集和测试集
    process_dataset('train')
    process_dataset('test')

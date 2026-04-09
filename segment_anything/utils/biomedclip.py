import os
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
from open_clip import get_tokenizer

# 只运行一次，获取 tokenizer 对象
tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

# 多次使用 tokenizer 编码文本
texts = [
    "a photo of a lung nodule",
    "a CT scan of skin lesions",
    "a healthy lung with no abnormality"
]

tokenized = tokenizer(texts)  # 支持列表或单条字符串
print(tokenized.shape)        # 输出: torch.Size([3, 77])（例如）

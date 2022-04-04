# 动漫头像生成项目
> 欢迎关注B站：https://space.bilibili.com/343147393

## 模型训练

### 数据集下载
小批量数据（6w张，411MB）：https://github.com/bchao1/Anime-Face-Dataset

大批量数据（14W张，10.9G）：https://www.kaggle.com/datasets/lukexng/animefaces-512x512

### 基于gan模型的训练
项目地址： https://github.com/chenyuntc/pytorch-book/tree/master/chapter07-AnimeGAN

首先下载数据集，然后把所有的图片放到`data/faces`目录下。

使用下面的命令开始进行训练
```bash
#安装依赖
pip install -r requirements.txt
#开始训模型
python main.py train --gpu --vis=False
```

> 注意：如果直接运行会报错，需要把 `main.py`的150-151行注释掉
> 
> 另外数据集里面其实有很多错误的图片，我们可以自己使用下面这个脚本自动删除错误的图片
```python
import os
from PIL import Image

if __name__ == '__main__':
    # 读取所有的文件
    for file in os.listdir("data/faces"):
        filename = "data/faces/%s" % file
        try:
            Image.open(filename)
        except:
            os.remove(filename)
            print("%s错误" % filename)
```

### 项目2（基于style-gan3）
https://github.com/NVlabs/stylegan3

## 项目运行

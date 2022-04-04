# 动漫头像生成项目
> 欢迎关注B站：https://space.bilibili.com/343147393

## 模型训练

### 数据集下载
小批量数据（6w张，411MB）：https://github.com/bchao1/Anime-Face-Dataset

大批量数据（14W张，10.9G）：https://www.kaggle.com/datasets/lukexng/animefaces-512x512

### 基于gan模型的训练
原项目地址： https://github.com/chenyuntc/pytorch-book/tree/master/chapter07-AnimeGAN

数据集地址：https://github.com/bchao1/Anime-Face-Dataset

首先下载数据集，然后把所有的图片放到`data/faces`目录下。

#### 训练模型

模型的一些参数都可以在train_gan.py里面进行配置

```bash
python train_gan.py
```
> 注意：数据集里面其实有很多错误的图片，我们可以自己使用下面这个脚本自动删除错误的图片
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
原项目地址： https://github.com/NVlabs/stylegan3

数据集地址：https://www.kaggle.com/datasets/lukexng/animefaces-512x512

#### 数据转换
默认stylegan3不支持我们前面的那个数据集，需要进行数据转换，转换代码如下
```bash
python arithmetic/stylegan3/dataset_tool.py --source=data/anime_face --dest=data/animation.zip
```

#### 开始训练
```bash
python train_style.py --outdir=data/out --cfg=stylegan3-t --data=data/animation.zip --gpus=1 --batch=8 --gamma=8.2 --mirror=1
```

## 项目运行
``bash
python main.py
``

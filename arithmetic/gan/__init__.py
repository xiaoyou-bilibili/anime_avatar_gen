# coding:utf8
import torch as t
import torchvision as tv
from arithmetic.gan.model import NetG, NetD

# 模型的配置信息，你训练时配置的多少，这里就需要修改成多少
nz = 100 # 噪声维度
ngf = 64 # 生成器feature map数
ndf = 64 # 判别器feature map数
# 模型路径
netd_path = 'model/gan/netd.pth'  # 预训练模型
netg_path = 'model/gan/netg.pth'
# 最后生成的路径
gen_img = 'web/static/gan_img.png'


def generate(
        gen_search_num:int=512,
        gen_num:int=64,
        gen_mean:int=0,
        gen_std:int=1,
    ):
    """
    随机生成动漫头像，并根据netd的分数选择较好的
    :param gen_search_num: 总生成图片数
    :param gen_num: 最终生成图片数
    :param gen_mean: 噪声的均值
    :param gen_std: 噪声的方差
    :return:
    """
    # 项目启动自动加载模型
    device = t.device('cuda')
    # 如果是使用CPU那么就把下面这个注释掉
    # device = t.device('cpu')
    # 加载我们的模型，
    netg, netd = NetG(ngf, nz).eval(), NetD(ndf).eval()
    map_location = lambda storage, loc: storage
    netd.load_state_dict(t.load(netd_path, map_location=map_location))
    netg.load_state_dict(t.load(netg_path, map_location=map_location))
    netd.to(device)
    netg.to(device)
    # 设置噪声信息，根据我们设置的总生成数来设置随机值
    noises = t.randn(gen_search_num, nz, 1, 1).normal_(gen_mean, gen_std)
    noises = noises.to(device)
    # 生成图片，并计算图片在判别器的分数
    fake_img = netg(noises)
    scores = netd(fake_img).detach()
    # 对我们的图片进行排序，选择较好图片
    indexs = scores.topk(gen_num)[1]
    result = []
    for ii in indexs:
        result.append(fake_img.data[ii])
    # 保存图片
    tv.utils.save_image(t.stack(result), gen_img, normalize=True, range=(-1, 1))


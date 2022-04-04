from flask import Flask, request, Response, render_template
from flask_cors import CORS
import json
from arithmetic.gan import generate


# 初始化flaskAPP
app = Flask(__name__)
# r'/*' 是通配符，让本服务器所有的URL 都允许跨域请求
# 允许跨域请求
CORS(app, resources=r'/*')


# 返回JSON字符串
def return_json(data):
    return Response(json.dumps(data, ensure_ascii=False), mimetype='application/json')


# 基于gan的动漫头像生成
@app.route('/gan/generate', methods=['POST'])
def detect_image():
    # 获取所有的参数
    data = request.form
    # 可以通过 request 的 args 属性来获取参数
    generate(
        gen_search_num=int(data.get("gen_search_num")),
        gen_num=int(data.get("gen_num")),
        gen_mean=int(data.get("gen_mean")),
        gen_std=int(data.get("gen_std")),
    )
    # 返回json类型字符串
    return return_json({
        "res": "/static/gan_img.png"
    })


# 基于图片的目标检测
@app.route('/yolov5/detect_video', methods=['POST'])
def detect_vide():
    # 获取请求的视频
    file = request.files['file']
    # 保存一下这个视频
    file.save("./web/static/detect.mp4")
    res = detect_video()
    # 返回json类型字符串
    return return_json(res)


# 主页显示HTML
@app.route('/', methods=['GET'])
def index():
    return render_template('content.html')

from arithmetic.gan import generate
from web import app

if __name__ == '__main__':
    # 加载模型
    # load_model()
    # 运行flask
    app.run(host='0.0.0.0', port=7001)
    # generate()
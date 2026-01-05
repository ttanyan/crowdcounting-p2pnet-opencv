# 使用支持 CUDA 12.2 的镜像，适配 T4 显卡和 Ubuntu 24.04 驱动
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

# 设置工作目录
WORKDIR /app

# 设置环境变量，解决安装时的交互问题
ENV DEBIAN_FRONTEND=noninteractive

# 更新并安装 Python 3.10 以及 OpenCV 所需的系统库
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 升级 pip 并安装依赖
COPY requirements.txt .
# T4 卡推荐安装 onnxruntime-gpu
RUN pip3 install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制项目代码（确保 weight 文件夹在当前目录下）
COPY . .

# 暴露端口
EXPOSE 8083

# 启动命令
CMD ["python3", "cuda_main.py"]
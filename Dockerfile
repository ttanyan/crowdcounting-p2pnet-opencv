# 使用官方 Python 3.8 slim 镜像（基于 Debian Bullseye）
FROM python:3.8-slim

# 设置环境变量：禁用缓冲、非交互安装、pip 使用清华源
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple/ \
    PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

# 替换 APT 源为清华，并安装 OpenCV 运行所需的最小系统依赖
RUN echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bullseye main" > /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bullseye-updates main" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian-security/ bullseye-security main" >> /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# 创建非 root 用户（安全最佳实践）
RUN useradd --create-home --shell /bin/bash appuser
USER appuser
WORKDIR /home/appuser/app

# 👇 关键：先复制依赖文件，单独安装 Python 包（利用 Docker 缓存）
COPY --chown=appuser:appuser requirements.txt .
RUN pip install  --upgrade pip && \
    pip install  --prefer-binary -r requirements.txt

# 复制应用代码和权重（这些常变动，放后面）
COPY --chown=appuser:appuser main.py .
COPY --chown=appuser:appuser weight/ ./weight/

# 创建图像目录
RUN mkdir -p imgs image_test

# 暴露端口
EXPOSE 8989

# 启动Flask API服务
ENV START_API_SERVICE=1
CMD ["python", "main.py"]
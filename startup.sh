#!/bin/bash

# 检查模型文件是否存在，如果不存在则提供下载说明
if [ ! -f "/app/weight/SHTechA.onnx" ]; then
    echo "警告: 模型文件 /app/weight/SHTechA.onnx 不存在"
    echo "请将模型文件复制到 weight/ 目录下"
    echo "或者从相应链接下载模型文件"
    mkdir -p /app/weight
fi

# 检查输入图像目录
if [ ! -d "/app/imgs" ]; then
    mkdir -p /app/imgs
    echo "已创建 imgs 目录，请将要处理的图像放入此目录"
fi

# 启动Flask应用
echo "启动人群计数API服务..."
exec python -c "from main import app; app.run(host='0.0.0.0', port=5000, debug=False)"
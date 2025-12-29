#!/bin/bash

# 检查是否已存在名为venv的虚拟环境
if [ -d "venv" ]; then
    echo "虚拟环境已存在，请先删除或重命名现有虚拟环境"
    exit 1
fi

echo "正在创建虚拟环境..."
python3 -m venv venv

echo "激活虚拟环境..."
source venv/bin/activate

echo "升级pip..."
pip install --upgrade pip

echo "安装依赖包..."
pip install -r requirements.txt

echo "环境设置完成！"

要激活虚拟环境，请运行: source venv/bin/activate
echo "要运行项目，请确保有相应的ONNX模型文件(SHTechA.onnx)和测试图片"
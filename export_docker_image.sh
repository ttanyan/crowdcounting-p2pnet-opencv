#!/bin/bash

# Docker镜像导出脚本
# 用于导出人群计数服务的Docker镜像为tar文件，便于离线部署

set -e  # 遇到错误时退出

# 查找当前构建的镜像
echo "查找人群计数服务镜像..."
IMAGE_NAME=""
for img in $(docker images --format "table {{.Repository}}:{{.Tag}}" | grep -E "(crowdcounting|p2pnet)" | grep -v "<none>"); do
    if [ -n "$img" ]; then
        IMAGE_NAME=$img
        break
    fi
done

# 如果没有找到特定命名的镜像，则使用默认的
if [ -z "$IMAGE_NAME" ]; then
    # 检查是否有通过docker-compose构建的镜像
    COMPOSE_IMAGE=$(docker images --format "json" | jq -r 'select(.Repository | contains("crowdcounting-p2pnet-opencv")) | .Repository + ":" + .Tag' | head -1)
    if [ -n "$COMPOSE_IMAGE" ] && [ "$COMPOSE_IMAGE" != "" ]; then
        IMAGE_NAME=$COMPOSE_IMAGE
    else
        IMAGE_NAME="crowdcounting-p2pnet:latest"
    fi
fi

# 生成导出文件名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPORT_FILE="crowdcounting_p2pnet_${IMAGE_NAME##*/}_${TIMESTAMP}.tar"

echo "将要导出镜像: $IMAGE_NAME"
echo "导出文件名: $EXPORT_FILE"

# 检查镜像是否存在
if docker inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo "开始导出镜像..."
    docker save "$IMAGE_NAME" -o "$EXPORT_FILE"
    echo "镜像已成功导出到: $EXPORT_FILE"
    echo "文件大小: $(du -h "$EXPORT_FILE" | cut -f1)"
else
    echo "错误: 镜像 '$IMAGE_NAME' 不存在"
    echo "请先构建镜像，例如："
    echo "  docker build -t crowdcounting-p2pnet:latest ."
    echo "或"
    echo "  docker-compose build"
    exit 1
fi

echo "导出完成！"
# 人群计数P2PNet OpenCV Docker版

基于P2PNet模型的人群计数服务，使用OpenCV和Flask封装为Docker容器化服务。

## 功能特性

- 通过Docker容器化部署
- RESTful API接口
- 支持base64格式图片输入
- 返回人群计数结果和处理后的图片
- 支持动态置信度阈值调整

## 架构

- 基于P2PNet模型的无人机视角人群计数
- 使用OpenCV进行推理
- Flask提供Web API服务
- Docker容器化部署

## API端点

### 健康检查
- `GET /health` - 检查服务状态

### 人群计数
- `POST /count_crowd` - 执行人群计数

请求格式：
```json
{
  "image_base64": "base64编码的图片字符串",
  "conf_threshold": 0.511
}
```

响应格式：
```json
{
  "success": true,
  "result_image_base64": "处理后图片的base64编码",
  "count": 人数,
  "processing_time": 处理时间
}
```

## Docker部署

### 构建镜像
```bash
docker build -t crowdcounting-p2pnet .
```

### 使用Docker Compose启动服务
```bash
docker-compose up -d
```

服务将在 `http://localhost:8080` 上运行（容器内部端口8989）。

### 手动运行容器
```bash
docker run -d \
  --name crowdcounting-api \
  -p 8080:8989 \
  -e START_API_SERVICE=1 \
  crowdcounting-p2pnet
```

## API使用示例

### 健康检查
```bash
curl http://localhost:8080/health
```

### 人群计数
```bash
curl -X POST http://localhost:8080/count_crowd \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "your_base64_encoded_image_string_here",
    "conf_threshold": 0.6
  }'
```

## 目录结构

- `main.py` - 主应用文件，包含P2PNet模型和Flask API
- `Dockerfile` - Docker构建配置
- `docker-compose.yml` - Docker Compose配置
- `requirements.txt` - Python依赖
- `weight/` - 模型文件目录
- `imgs/` - 输入图片目录
- `image_test/` - 输出图片目录

## 环境变量

- `START_API_SERVICE=1` - 启动Flask API服务而不是命令行模式

## 注意事项

1. 确保 `weight/` 目录包含 `SHTechA.onnx` 模型文件
2. 服务启动后会自动加载模型到内存
3. 首次请求可能需要一些时间进行初始化
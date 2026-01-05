# 人群计数API使用说明

## 接口功能
- 接收base64编码的图片和置信度参数
- 返回处理后带有检测点的base64图片
- 同时返回检测人数和处理时间

## 直接API调用（推荐）

### 简化版API（最简单）
```python
from simple_api import count_crowd_api, process_image_api
import base64

# 方法1: 获取完整结果
with open('your_image.jpg', 'rb') as f:
    image_data = f.read()
    image_base64 = base64.b64encode(image_data).decode('utf-8')

result = count_crowd_api(
    image_base64=image_base64,
    conf_threshold=0.6,           # 置信度参数
    model_path='weight/SHTechA.onnx',  # 模型路径
    use_gpu=False                 # 是否使用GPU
)

if result['success']:
    processed_image_base64 = result['result_image_base64']  # 处理后的图片base64
    count = result['count']                                  # 检测人数
    processing_time = result['processing_time']              # 处理时间
else:
    print(f"处理失败: {result['error']}")

# 方法2: 只获取处理后的图片
result_image_base64 = process_image_api(
    image_base64=image_base64,
    conf_threshold=0.6
)
```

### 完整版API
```python
from api import CrowdCountingAPI
import base64

# 初始化API
api = CrowdCountingAPI(model_path='weight/SHTechA.onnx', conf_threshold=0.511)

# 处理base64图片
result_info = api.process_image_base64_with_info(
    image_base64, 
    conf_threshold=0.6  # 可以动态调整置信度
)

# result_info 包含: {'image_base64': ..., 'count': 人数, 'processing_time': 处理时间}
```

## Web API使用

### 启动服务
```bash
# 使用Docker Compose启动
docker-compose up

# 或直接运行
python web_api.py
```

### API端点
- `POST /count_crowd` - 人群计数接口
- `GET /health` - 健康检查接口

### 请求示例
```bash
curl -X POST http://localhost:5000/count_crowd \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "your_base64_encoded_image_string_here",
    "conf_threshold": 0.6
  }'
```

## Docker部署
```bash
# 构建镜像
docker build -t crowdcounting-p2pnet .

# 运行容器
docker run -p 5000:8989 \
  -v $(pwd)/weight:/app/weight \
  -v $(pwd)/imgs:/app/imgs \
  crowdcounting-p2pnet

# 或使用docker-compose
docker-compose up
```

## Docker容器化部署（新方法 - 推荐）

### 构建和运行
```bash
# 构建镜像
docker build -t crowdcounting-p2pnet .

# 使用Docker Compose启动服务
docker-compose up -d

# 服务将在 http://localhost:8080 上运行（容器内部端口8989）
```

### API端点
- `GET /health` - 健康检查
- `POST /count_crowd` - 人群计数

### 测试API
```bash
# 健康检查
curl http://localhost:8080/health

# 人群计数（需要提供base64编码的图片）
curl -X POST http://localhost:8080/count_crowd \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "your_base64_encoded_image_string_here",
    "conf_threshold": 0.6
  }'
```

## 环境要求
- Python 3.8+
- 安装依赖: `pip install -r requirements.txt`
- 模型文件: `weight/SHTechA.onnx`
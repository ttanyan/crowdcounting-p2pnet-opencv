import argparse
import cv2
import numpy as np
import os
import datetime
import random
import time
import base64
from flask import Flask, request, jsonify


class AnchorPoints():
    def __init__(self, pyramid_levels=None, strides=None, row=2, line=2):
        # 无人机视角下，row/line 设为 2x2 通常比 3x3 在小目标上更平衡
        self.pyramid_levels = pyramid_levels if pyramid_levels is not None else [3]
        self.strides = strides if strides is not None else [2 ** x for x in self.pyramid_levels]
        self.row = row
        self.line = line

    def generate_anchor_points(self, stride=8, row=2, line=2):
        # 矢量化生成 anchor 偏移
        row_step = stride / row
        line_step = stride / line
        shift_x = (np.arange(1, line + 1) - 0.5) * line_step - stride / 2
        shift_y = (np.arange(1, row + 1) - 0.5) * row_step - stride / 2
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        return np.stack([shift_x.ravel(), shift_y.ravel()], axis=1)

    def shift(self, shape, stride, anchor_points):
        # 优化后的 shift 函数，减少内存拷贝
        y, x = np.mgrid[0:shape[0], 0:shape[1]]
        shifts = np.stack([(x.ravel() + 0.5) * stride, (y.ravel() + 0.5) * stride], axis=1)

        # 使用 broadcasting 快速计算所有 anchor 坐标
        all_anchors = shifts[:, np.newaxis, :] + anchor_points[np.newaxis, :, :]
        return all_anchors.reshape(-1, 2)

    def __call__(self, image_shape_hw):
        # 传入 HW 即可，无需整个图像 blob
        all_anchor_points = []
        for idx, p in enumerate(self.pyramid_levels):
            stride = 2 ** p
            # 计算当前特征层的大小
            f_h, f_w = (image_shape_hw[0] + stride - 1) // stride, (image_shape_hw[1] + stride - 1) // stride
            anchors = self.generate_anchor_points(stride, row=self.row, line=self.line)
            shifted = self.shift((f_h, f_w), self.strides[idx], anchors)
            all_anchor_points.append(shifted)

        return np.concatenate(all_anchor_points, axis=0).astype(np.float32)


class P2PNet():
    def __init__(self, model_path, conf_threshold=0.511, use_gpu=False):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型文件: {model_path}")

        self.net = cv2.dnn.readNet(model_path)
        if use_gpu:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        self.conf_threshold = conf_threshold
        # ImageNet 标准归一化参数
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        # SHTechA 预训练模型通常对应 pyramid_levels=[3], row=2, line=2
        self.anchor_gen = AnchorPoints(pyramid_levels=[3], row=2, line=2)
        
        # 预加载模型到内存（在初始化时完成）
        self.model_loaded = True

    def _sigmoid(self, x):
        # 优化的sigmoid函数，避免数值溢出
        x = np.clip(x, -500, 500)  # 限制范围避免溢出
        return np.where(x >= 0, 
                        1 / (1 + np.exp(-x)), 
                        np.exp(x) / (1 + np.exp(x)))

    def detect(self, srcimg):
        h, w = srcimg.shape[:2]
        # 128 对齐，防止网络下采样时的维度不匹配
        new_w, new_h = (w // 128) * 128, (h // 128) * 128

        # 预处理优化：直接在 blobFromImage 中完成归一化会更快
        # 但由于 OpenCV 默认只支持减均值，不支持除以标准差，所以仍手动处理或在网络中集成
        blob_img = cv2.resize(srcimg, (new_w, new_h))
        blob_img = cv2.cvtColor(blob_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob_img = (blob_img - self.mean) / self.std

        blob = cv2.dnn.blobFromImage(blob_img)

        self.net.setInput(blob)
        # 获取输出节点：pred_logits (置信度), pred_points (偏移量)
        out_names = self.net.getUnconnectedOutLayersNames()
        preds = self.net.forward(out_names)

        # 解析输出
        # 不同导出方式输出顺序可能不同，需通过 shape 判定
        logits = preds[0] if preds[0].shape[-1] == 1 else preds[1]
        offset = preds[1] if preds[1].shape[-1] == 2 else preds[0]

        # 获取 Anchor Points
        anchors = self.anchor_gen((new_h, new_w))

        # 计算最终坐标
        scores = self._sigmoid(logits.flatten())
        points = offset.reshape(-1, 2) + anchors

        # 过滤
        keep = scores > self.conf_threshold
        final_scores = scores[keep]
        final_points = points[keep]

        # 映射回原图尺寸
        final_points[:, 0] *= (w / new_w)
        final_points[:, 1] *= (h / new_h)

        return final_scores, final_points
        
    def inference(self, srcimg, conf_threshold=None):
        """
        执行推理并返回检测结果
        
        Args:
            srcimg: 输入图像
            conf_threshold: 置信度阈值，如果为None则使用初始化时的阈值
        
        Returns:
            tuple: (scores, points) - 检测分数和坐标点
        """
        if conf_threshold is None:
            conf_threshold = self.conf_threshold
            
        h, w = srcimg.shape[:2]
        # 128 对齐，防止网络下采样时的维度不匹配
        new_w, new_h = (w // 128) * 128, (h // 128) * 128

        # 预处理优化：直接在 blobFromImage 中完成归一化会更快
        # 但由于 OpenCV 默认只支持减均值，不支持除以标准差，所以仍手动处理或在网络中集成
        blob_img = cv2.resize(srcimg, (new_w, new_h))
        blob_img = cv2.cvtColor(blob_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob_img = (blob_img - self.mean) / self.std

        blob = cv2.dnn.blobFromImage(blob_img)

        self.net.setInput(blob)
        # 获取输出节点：pred_logits (置信度), pred_points (偏移量)
        out_names = self.net.getUnconnectedOutLayersNames()
        preds = self.net.forward(out_names)

        # 解析输出
        # 不同导出方式输出顺序可能不同，需通过 shape 判定
        logits = preds[0] if preds[0].shape[-1] == 1 else preds[1]
        offset = preds[1] if preds[1].shape[-1] == 2 else preds[0]

        # 获取 Anchor Points
        anchors = self.anchor_gen((new_h, new_w))

        # 计算最终坐标
        scores = self._sigmoid(logits.flatten())
        points = offset.reshape(-1, 2) + anchors

        # 过滤 - 使用更快的过滤方式
        keep = scores > conf_threshold
        final_scores = scores[keep]
        final_points = points[keep]

        # 映射回原图尺寸
        final_points[:, 0] *= (w / new_w)
        final_points[:, 1] *= (h / new_h)

        return final_scores, final_points

    def draw_and_count(self, srcimg, conf_threshold=None):
        """
        执行检测并在图像上绘制结果
        
        Args:
            srcimg: 输入图像
            conf_threshold: 置信度阈值，如果为None则使用初始化时的阈值
        
        Returns:
            tuple: (结果图像, 处理时间)
        """
        if conf_threshold is None:
            conf_threshold = self.conf_threshold
            
        # 记录开始时间
        start_time = time.time()
        
        # 检测
        scores, points = self.inference(srcimg, conf_threshold)

        # 绘制结果 - 使用批量操作提高效率
        result_img = srcimg.copy()
        
        # 使用numpy向量化操作绘制所有点
        if len(points) > 0:
            pts = points.astype(int)
            for pt in pts:
                cv2.circle(result_img, (pt[0], pt[1]), 3, (0, 255, 0), -1)

        # 显示计数
        cv2.putText(result_img, f"Count: {len(points)}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        # 显示置信度阈值
        cv2.putText(result_img, f"Confidence: {conf_threshold}", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # 记录结束时间并计算耗时
        end_time = time.time()
        processing_time = end_time - start_time
        
        return result_img, processing_time

    def process_frame_batch(self, frames, conf_threshold=None):
        """
        批量处理多帧图像
        
        Args:
            frames: 图像列表
            conf_threshold: 置信度阈值
        
        Returns:
            list: (处理后的图像, 耗时) 元组列表
        """
        results = []
        for frame in frames:
            result, proc_time = self.draw_and_count(frame, conf_threshold)
            results.append((result, proc_time))
        return results


def process_base64_image(image_base64, model_path='weight/SHTechA.onnx', conf_threshold=0.511, use_gpu=False):
    """
    处理base64图片的API接口 - 接收图片base64和置信度参数，返回计算好的base64图片
    
    Args:
        image_base64 (str): 输入图片的base64编码字符串
        model_path (str): 模型文件路径
        conf_threshold (float): 置信度阈值
        use_gpu (bool): 是否使用GPU
    
    Returns:
        dict: 包含处理后的图片base64、人数和处理时间的字典
    """
    # 解码base64图片
    image_data = base64.b64decode(image_base64)
    np_arr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("无法解码输入的base64图片")
    
    # 初始化检测器
    detector = P2PNet(model_path, conf_threshold=conf_threshold, use_gpu=use_gpu)
    
    # 处理图像
    result_img, processing_time = detector.draw_and_count(img, conf_threshold)
    
    # 获取检测人数
    scores, points = detector.inference(img, conf_threshold)
    count = len(points)
    
    # 将处理后的图片编码为base64
    _, buffer = cv2.imencode('.jpg', result_img)
    result_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        'result_image_base64': result_base64,
        'count': count,
        'processing_time': processing_time
    }


# 创建Flask应用
app = Flask(__name__)

# 在应用启动时预加载模型
model = None

@app.before_first_request
def load_model():
    global model
    model = P2PNet('weight/SHTechA.onnx', use_gpu=False)

@app.route('/count_crowd', methods=['POST'])
def count_crowd_api():
    """
    人群计数API接口
    接收JSON格式的请求，包含image_base64和conf_threshold字段
    返回处理后的图片base64编码
    """
    try:
        data = request.get_json()
        
        if not data or 'image_base64' not in data:
            return jsonify({
                'error': '缺少image_base64字段'
            }), 400
        
        image_base64 = data['image_base64']
        conf_threshold = data.get('conf_threshold', 0.511)
        
        if conf_threshold is not None:
            try:
                conf_threshold = float(conf_threshold)
            except ValueError:
                return jsonify({
                    'error': 'conf_threshold必须是数字'
                }), 400
        
        # 处理图片
        result = process_base64_image(
            image_base64=image_base64,
            model_path='weight/SHTechA.onnx',
            conf_threshold=conf_threshold,
            use_gpu=False
        )
        
        return jsonify({
            'success': True,
            'result_image_base64': result['result_image_base64'],
            'count': result['count'],
            'processing_time': result['processing_time']
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """
    健康检查接口
    """
    return jsonify({
        'status': 'healthy',
        'model_path': 'weight/SHTechA.onnx',
        'default_conf_threshold': 0.511
    })


if __name__ == '__main__':
    # 检查是否需要启动API服务（通过环境变量）
    if os.environ.get('START_API_SERVICE') == '1':
        # 启动Flask API服务
        app.run(host='0.0.0.0', port=8989, debug=False)
    else:
        # 运行命令行模式
        parser = argparse.ArgumentParser()
        parser.add_argument('--imgpath', default='imgs/img_5.png')
        parser.add_argument('--onnx_path', default='weight/SHTechA.onnx')
        parser.add_argument('--gpu', action='store_true', help="是否使用 CUDA")
        args = parser.parse_args()

        # 初始化
        detector = P2PNet(args.onnx_path, use_gpu=args.gpu)

        # 读取图像
        frame = cv2.imread(args.imgpath)
        if frame is None:
            print("tu")
            exit()

        # 记录开始时间
        start_time = time.time()

        # 检测
        scores, points = detector.detect(frame)

        # 绘制结果
        for pt in points:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)

        # 显示计数
        cv2.putText(frame, f"Count: {len(points)}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        # 计算并显示处理时间
        end_time = time.time()
        processing_time = end_time - start_time
        cv2.putText(frame, f"Time: {processing_time:.3f}s", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        print(f"检测完成，人数: {len(points)}")
        print(f"处理耗时: {processing_time:.3f}秒")

        # 创建保存目录
        output_dir = "image_test"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 生成时间戳加随机数的文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        random_num = random.randint(1000, 9999)
        filename = f"{output_dir}/result_{timestamp}_{random_num}.jpg"
        
        # 保存结果图片
        cv2.imwrite(filename, frame)
        print(f"检测结果已保存到: {filename}")

        # 窗口展示优化（针对大图缩小展示）
        h, w = frame.shape[:2]
        show_w = 1280
        show_h = int(h * (show_w / w))
        cv2.namedWindow('P2PNet Drone View', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('P2PNet Drone View', show_w, show_h)
        cv2.imshow('P2PNet Drone View', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
else:
    # 当作为模块导入时
    pass
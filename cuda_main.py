import cv2
import numpy as np
import onnxruntime as ort
import io
from flask import Flask, request, send_file, jsonify

app = Flask(__name__)


class P2PNetDetector:
    def __init__(self, model_path):
        # 如果使用 GPU，请确保安装了 onnxruntime-gpu
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def process(self, image_bytes, conf):
        # 解码图片
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None, 0

        h, w = img.shape[:2]
        nh, nw = (h // 128) * 128, (w // 128) * 128

        # 预处理
        blob = cv2.resize(img, (nw, nh))
        blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = (blob - self.mean) / self.std
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

        # 推理
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: blob})
        logits = outputs[0].flatten()
        points = outputs[1].reshape(-1, 2)

        # 锚点逻辑
        stride = 8
        f_h, f_w = nh // stride, nw // stride
        y, x = np.mgrid[0:f_h, 0:f_w]
        anchors_base = np.stack([x.flatten(), y.flatten()], axis=1) * stride + 0.5 * stride
        anchors = np.tile(anchors_base[:, np.newaxis, :], (1, 4, 1)).reshape(-1, 2)

        # 过滤 (使用传入的置信度)
        scores = 1 / (1 + np.exp(-logits))
        mask = scores > conf
        final_pts = points[mask] + anchors[mask]

        # 坐标映射
        final_pts[:, 0] *= (w / nw)
        final_pts[:, 1] *= (h / nh)

        # 绘制
        for p in final_pts:
            cv2.circle(img, (int(p[0]), int(p[1])), 4, (0, 255, 0), -1)

        cv2.putText(img, f"Count: {len(final_pts)} Conf: {conf}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        _, buffer = cv2.imencode('.jpg', img)
        return buffer.tobytes(), len(final_pts)


# 初始化模型 (确保路径在 Docker 容器内正确)
detector = P2PNetDetector('weight/SHTechA.onnx')


@app.route('/predict', methods=['POST'])
def predict():
    # 1. 检查图片是否存在
    if 'image' not in request.files:
        return jsonify({"error": "No image field provided"}), 400

    # 2. 获取置信度参数 (如果没传则默认 0.5)
    try:
        conf_threshold = float(request.form.get('conf', 0.5))
    except ValueError:
        return jsonify({"error": "Invalid confidence value"}), 400

    file = request.files['image']
    img_bytes = file.read()

    # 3. 处理图片
    result_img_bytes, count = detector.process(img_bytes, conf_threshold)

    if result_img_bytes is None:
        return jsonify({"error": "Failed to process image"}), 500

    # 4. 返回结果图片
    return send_file(
        io.BytesIO(result_img_bytes),
        mimetype='image/jpeg'
    )


if __name__ == '__main__':
    # 注意这里端口改为你日志中的 8083
    app.run(host='0.0.0.0', port=8083)
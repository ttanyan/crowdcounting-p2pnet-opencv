import cv2
import numpy as np
import onnxruntime as ort
import io
import time
from flask import Flask, request, send_file, jsonify

app = Flask(__name__)


class P2PNetCPU:
    def __init__(self, model_path):
        # 强制使用 CPU
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def detect_and_draw(self, img, conf=0.5):
        h, w = img.shape[:2]
        # P2PNet 128 对齐
        nh, nw = (h // 128) * 128, (w // 128) * 128

        # 1. 预处理
        blob = cv2.resize(img, (nw, nh))
        blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = (blob - self.mean) / self.std
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

        # 2. 推理
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: blob})
        logits = outputs[0].flatten()
        points = outputs[1].reshape(-1, 2)

        # 3. 锚点生成 (8倍下采样)
        stride = 8
        f_h, f_w = nh // stride, nw // stride
        y, x = np.mgrid[0:f_h, 0:f_w]
        anchors_base = np.stack([x.flatten(), y.flatten()], axis=1) * stride + 0.5 * stride
        anchors = np.tile(anchors_base[:, np.newaxis, :], (1, 4, 1)).reshape(-1, 2)

        # 4. 激活与过滤
        scores = 1 / (1 + np.exp(-logits))
        mask = scores > conf
        final_pts = points[mask] + anchors[mask]

        # 5. 映射回原图并绘制
        final_pts[:, 0] *= (w / nw)
        final_pts[:, 1] *= (h / nh)

        for p in final_pts:
            cv2.circle(img, (int(p[0]), int(p[1])), 4, (0, 255, 0), -1)

        cv2.putText(img, f"Count: {len(final_pts)}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        return img, len(final_pts)


# 初始化模型
detector = P2PNetCPU('weight/SHTechA.onnx')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    conf = float(request.form.get('conf', 0.53))  # 默认使用你代码里的 0.53

    # 读取图片字节流
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    # 检测并绘制
    result_img, count = detector.detect_and_draw(img, conf=conf)

    # 将结果图转回字节流
    _, buffer = cv2.imencode('.jpg', result_img)
    return send_file(io.BytesIO(buffer), mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8083)
import requests
import base64
import json
import os

# 读取图片并转换为base64编码
def get_image_base64(image_path):
    with open(image_path, 'rb') as f:
        image_data = f.read()
        return base64.b64encode(image_data).decode('utf-8')

# 将base64图片数据保存到文件
def save_base64_image(base64_data, output_path):
    """
    将base64编码的图片数据保存到指定路径
    :param base64_data: base64编码的图片数据（可能包含data前缀，需要处理）
    :param output_path: 输出文件路径
    """
    # 如果base64字符串包含data前缀（如 "data:image/jpeg;base64,"），则去除它
    if ',' in base64_data:
        base64_data = base64_data.split(',')[1]
    
    # 解码base64数据并保存为图片
    image_data = base64.b64decode(base64_data)
    with open(output_path, 'wb') as f:
        f.write(image_data)
    print(f"处理后的图片已保存到: {output_path}")

# 测试人群计数接口
def test_count_crowd():
    # 图片路径（请确保图片文件存在）
    image_path = "imgs/test11.jpg"
    
    try:
        # 获取图片的base64编码
        image_base64 = get_image_base64(image_path)
        
        # API端点
        url = "http://localhost:8080/count_crowd"
        
        # 请求数据
        payload = {
            "image_base64": image_base64,
            "conf_threshold": 0.52  # 可以根据需要调整置信度阈值
        }
        
        # 发送POST请求
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, data=json.dumps(payload), headers=headers)
        
        # 打印响应
        print("API响应:")
        print(json.dumps(response.json(), indent=2))
        
        # 检查响应是否成功
        if response.status_code == 200 and response.json().get('success'):
            print("\n✅ 人群计数测试成功！")
            print(f"检测人数: {response.json()['count']}")
            print(f"处理时间: {response.json()['processing_time']:.3f}秒")
            
            # 获取返回的处理后图片的base64数据
            result_image_base64 = response.json().get('result_image_base64')
            if result_image_base64:
                # 确保image_test目录存在
                os.makedirs('image_test', exist_ok=True)
                
                # 生成输出文件路径
                import time
                timestamp = int(time.time())
                output_path = f"image_test/result_img_{timestamp}.jpg"
                
                # 保存图片
                save_base64_image(result_image_base64, output_path)
            else:
                print("⚠️  API响应中未包含处理后的图片数据")
        else:
            print("\n❌ 人群计数测试失败！")
            print(f"错误信息: {response.json().get('error', '未知错误')}")
            
    except FileNotFoundError:
        print(f"❌ 找不到图片文件: {image_path}")
        print("请确保图片文件存在于项目根目录中，并命名为 'test01.png'")
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")

if __name__ == "__main__":
    print("开始测试人群计数接口...")
    test_count_crowd()
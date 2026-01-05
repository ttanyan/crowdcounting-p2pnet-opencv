import requests
import base64
import json
import os

# 创建一个简单的测试图片（如果存在的话）
# 如果没有测试图片，我们可以先测试健康检查接口
def test_health_check():
    url = "http://localhost:8080/health"
    try:
        response = requests.get(url)
        print("健康检查响应:", response.json())
        return True
    except Exception as e:
        print(f"健康检查失败: {e}")
        return False

def test_crowd_counting():
    # 这里需要一个测试图片的base64编码
    # 如果没有测试图片，我们先跳过这个测试
    print("要测试人群计数功能，请提供一个图片的base64编码")
    return True

def test_crowd_counting_with_image(image_path, conf_threshold=0.5):
    """
    测试人群计数API并将返回的图片保存到image_test目录
    :param image_path: 本地图片路径
    :param conf_threshold: 置信度阈值
    :return: 是否测试成功
    """
    # 读取本地图片并转换为base64
    try:
        with open(image_path, 'rb') as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        print(f"读取图片失败: {e}")
        return False
    
    # API请求
    url = "http://localhost:8080/process"
    headers = {'Content-Type': 'application/json'}
    data = {
        'image': img_base64,
        'conf_threshold': conf_threshold
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            result = response.json()
            if 'image' in result:
                # 解码base64图片数据
                returned_img_base64 = result['image']
                returned_img_data = base64.b64decode(returned_img_base64)
                
                # 确保image_test目录存在
                os.makedirs('image_test', exist_ok=True)
                
                # 生成输出文件名
                base_name = os.path.basename(image_path)
                name, ext = os.path.splitext(base_name)
                output_path = f"image_test/{name}_result_{conf_threshold}.jpg"
                
                # 保存图片
                with open(output_path, 'wb') as output_file:
                    output_file.write(returned_img_data)
                
                print(f"处理结果已保存到: {output_path}")
                print(f"检测到的人数: {result.get('count', 'N/A')}")
                print(f"处理时间: {result.get('processing_time', 'N/A')}秒")
                return True
            else:
                print("API响应中没有图片数据")
                return False
        else:
            print(f"API请求失败，状态码: {response.status_code}")
            print(f"响应内容: {response.text}")
            return False
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        return False

def test_with_sample_image():
    """
    使用示例图片进行测试（如果存在）
    """
    sample_images = [
        'imgs/test.jpg',
        'imgs/test.png',
        'imgs/sample.jpg',
        'imgs/sample.png'
    ]
    
    for img_path in sample_images:
        if os.path.exists(img_path):
            print(f"使用图片 {img_path} 进行测试...")
            success = test_crowd_counting_with_image(img_path, conf_threshold=0.5)
            if success:
                return True
            else:
                print(f"使用图片 {img_path} 测试失败，尝试下一张...")
    
    print("未找到可用的测试图片，您可以将图片放在imgs目录下并命名为test.jpg或sample.jpg")
    return False

if __name__ == "__main__":
    print("测试API服务...")
    if test_health_check():
        print("API健康检查通过")
        
        # 尝试使用示例图片进行测试
        print("\n开始人群计数功能测试...")
        test_with_sample_image()
        
        print("\nAPI服务测试完成")
    else:
        print("API健康检查失败")
        print("请确保Docker容器正在运行，命令: docker-compose up")
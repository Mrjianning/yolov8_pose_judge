"""
Author: lvjianing
Email: 1046016768@qq.com
Date: June 22, 2024
"""

import torch
import cv2 as cv
import numpy as np
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from ultralytics.engine.results import Results
import copy
from PIL import Image, ImageDraw, ImageFont
import math

class YOLOv8Pose:
    def __init__(self, model_path, device='cpu', conf=0.25, iou=0.7):
        self.device = device
        self.conf = conf
        self.iou = iou
        self.model = self.load_model(model_path)
        self.letterbox = LetterBox([640, 640], auto=True, stride=32)
    
    def load_model(self, model_path):
        ckpt = torch.load(model_path, map_location=self.device)
        model = ckpt['model'].to(self.device).float()
        model.eval()
        return model
    
    def preprocess(self, img_path):
        im = cv.imread(img_path)
        im = [im]
        orig_imgs = copy.deepcopy(im)
        im = [self.letterbox(image=x) for x in im]
        im = im[0][None]
        im = im[..., ::-1].transpose((0, 3, 1, 2))
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im)
        img = im.to(self.device)
        img = img.float()
        img /= 255
        return img, orig_imgs
    
    def infer(self, img):
        preds = self.model(img)
        prediction = ops.non_max_suppression(preds, self.conf, self.iou, agnostic=False, max_det=300, classes=None, nc=len(self.model.names))
        return prediction
    
    def postprocess(self, prediction, orig_imgs, img_shape, img_path):
        results = []
        for i, pred in enumerate(prediction):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            shape = orig_img.shape
            pred[:, :4] = ops.scale_boxes(img_shape, pred[:, :4], shape).round()
            pred_kpts = pred[:, 6:].view(len(pred), *self.model.kpt_shape) if len(pred) else pred[:, 6:]
            pred_kpts = ops.scale_coords(img_shape, pred_kpts, shape)
            
            results.append(
                Results(orig_img=orig_img,
                        path=img_path,
                        names=self.model.names,
                        boxes=pred[:, :6],
                        keypoints=pred_kpts))
        return results
    
    def plot_results(self, results, img):
        plot_args = {'line_width': None, 'boxes': True, 'conf': True, 'labels': True}
        plot_args['im_gpu'] = img[0]
        result = results[0]
        plotted_img = result.plot(**plot_args)
        return plotted_img
    
    def detect(self, img_path):
        img, orig_imgs = self.preprocess(img_path)
        prediction = self.infer(img)
        results = self.postprocess(prediction, orig_imgs, img.shape[2:], img_path)
        return results
    
    def show_results(self, results, img_path):
        img, _ = self.preprocess(img_path)
        plotted_img = self.plot_results(results, img)
        
        cv.imshow('plotted_img', plotted_img)
        cv.waitKey(0)
        cv.destroyAllWindows()
    
    import math

    def determine_pose(self, keypoints):
        """
        通过关键点判断人体的姿态
        :param keypoints: 关键点坐标和置信度，格式为 [(x1, y1, conf1), (x2, y2, conf2), ...]
        :return: 姿态类别： 'stand'（站立）, 'walk'（行走）, 'jump'（跳跃）, 'unknown'（未知）
        """
        # 如果关键点数量少于17个，直接返回'unknown'
        if len(keypoints) < 17:
            return 'unknown'

        # 定义一些关键点的索引
        left_ankle = keypoints[15]   # 左脚踝
        right_ankle = keypoints[16]  # 右脚踝
        left_knee = keypoints[13]    # 左膝盖
        right_knee = keypoints[14]   # 右膝盖
        left_hip = keypoints[11]     # 左髋关节
        right_hip = keypoints[12]    # 右髋关节

        # 获取关键点的坐标和置信度
        left_ankle_x, left_ankle_y, left_ankle_conf = left_ankle
        right_ankle_x, right_ankle_y, right_ankle_conf = right_ankle
        left_knee_x, left_knee_y, left_knee_conf = left_knee
        right_knee_x, right_knee_y, right_knee_conf = right_knee
        left_hip_x, left_hip_y, left_hip_conf = left_hip
        right_hip_x, right_hip_y, right_hip_conf = right_hip

        # 计算膝盖关节之间的角度
        def calculate_knee_angle(knee, hip, ankle):
            # 计算两个向量的夹角
            vector1 = (knee[0] - hip[0], knee[1] - hip[1])
            vector2 = (ankle[0] - knee[0], ankle[1] - knee[1])

            dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
            magnitude1 = math.sqrt(vector1[0]**2 + vector1[1]**2)
            magnitude2 = math.sqrt(vector2[0]**2 + vector2[1]**2)

            if magnitude1 * magnitude2 == 0:
                return 0

            cos_theta = dot_product / (magnitude1 * magnitude2)
            angle_rad = math.acos(cos_theta)
            angle_deg = math.degrees(angle_rad)
            return angle_deg

        # 简单的姿态判断逻辑
        if left_ankle_conf > 0.5 and right_ankle_conf > 0.5:
            # 在 OpenCV 坐标系中，y 坐标值越大，位置越低

            # 计算左脚角度
            knee_angle_left = calculate_knee_angle(left_knee, left_hip, left_ankle)
            # 计算右脚角度
            knee_angle_right = calculate_knee_angle(right_knee, right_hip, right_ankle)
            print("左脚角度,",knee_angle_left)
            print("右脚角度,",knee_angle_right)

            # 计算脚踝之间的距离
            ankle_distance = abs(left_ankle_x - right_ankle_x)
            # 计算膝盖之间的距离
            knee_distance = abs(left_knee_x - right_knee_x)
            print("脚踝之间的距离,",ankle_distance)
            print("膝盖之间的距离,",knee_distance)

            # 判断逻辑
            flag_0=left_ankle_y > left_knee_y > left_hip_y and right_ankle_y > right_knee_y > right_hip_y
            flag_1=knee_angle_left < 30 and knee_angle_right < 30
            flag_2=ankle_distance > knee_distance*1.5
            flag_3=ankle_distance < knee_distance*1.5

            # 如果左脚踝和右脚踝的 y 坐标都大于膝盖和髋关节的 y 坐标，并且膝盖之间的角度小于一定阈值，判定为站立
            if (flag_0 and flag_1 and flag_3):
                return '站立'
            
            # 如果脚踝之间的距离大于膝盖之间的距离，判定为行走
            if flag_2:
                return '行走'

            # 如果左脚踝和右脚踝的 y 坐标都大于髋关节的 y 坐标，并且膝盖之间的角度大于一定阈值，判定为跳跃
            if left_ankle_y > left_hip_y and right_ankle_y > right_hip_y:
                if knee_angle_left > 60 or knee_angle_right > 60:  # 设置角度阈值为160度
                    return '跳跃'
        
        # 如果以上条件都不满足，返回'unknown'
        return '未知'


    def get_results_as_dicts(self, results):
        results_list = []
        for result in results:
            for i, box in enumerate(result.boxes.data):
                cls_id = result.boxes.cls[i].item()       # 获取类别ID
                class_name = result.names.get(cls_id)
                confidence = result.boxes.conf[i].item()  # 获取置信度
                bbox = box[:4].tolist()                   # 获取边框坐标

                # 准备关键点信息
                keypoints = []
                if result.keypoints.has_visible:
                    for kpt in result.keypoints.data:
                        for point in kpt:
                            x, y, conf = point.tolist()
                            keypoints.append((x, y, conf))

                # 判断姿态
                pose = self.determine_pose(keypoints)

                result_dict = {
                    "class_name": class_name,
                    "boxes": bbox,
                    "confidence": confidence,
                    "keypoints": keypoints,
                    "pose": pose  # 增加姿态信息
                }
                results_list.append(result_dict)
        
        return results_list

# 关键点的连接关系（COCO数据集的例子）
skeleton = [
    (0, 1), (0, 2), (1, 3), (2, 4),(3, 5),(4, 6),              # 头部到身体
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),                   # 身体到手臂
    (5, 11), (6, 12), (11, 12),                                # 躯干
    (11, 13), (12, 14), (13, 15), (14, 16)                     # 躯干到腿
]

# opencv绘制结果
def draw_results(image, results):
    height, width, _ = image.shape
    scale_factor = height / 500  # 假设基准高度为500像素

    line_thickness = max(1, int(2 * scale_factor))  # 线条粗细至少为1
    font_scale = max(0.5, 0.5 * scale_factor)  # 字体缩放至少为0.5
    circle_radius = max(1, int(3 * scale_factor))  # 圆点半径至少为1

    for res in results:
        # 绘制边框和类别名
        bbox = res['boxes']
        confidence = res['confidence']
        class_name = res['class_name']

        start_point = (int(bbox[0]), int(bbox[1]))
        end_point = (int(bbox[2]), int(bbox[3]))
        cv.rectangle(image, start_point, end_point, (255, 0, 0), line_thickness)

        label = f"{class_name}: {confidence:.2f}"
        cv.putText(image, label, (start_point[0], start_point[1] - 10), cv.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), line_thickness)

        # 绘制关键点
        keypoints = [(int(kpt[0]), int(kpt[1]), kpt[2]) for kpt in res['keypoints']]
        for idx, (x, y, conf) in enumerate(keypoints):
            if conf > 0.5:
                cv.circle(image, (x, y), circle_radius, (0, 255, 0), -1)
                cv.putText(image, str(idx), (x + 5, y + 5), cv.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), line_thickness)
        
        # 绘制骨架
        for start_idx, end_idx in skeleton:
            start_point = keypoints[start_idx]
            end_point = keypoints[end_idx]
            if start_point[2] > 0.5 and end_point[2] > 0.5:
                cv.line(image, (start_point[0], start_point[1]), (end_point[0], end_point[1]), (0, 255, 0), line_thickness)

        # 显示姿态信息
        pose_label = f"姿态: {res['pose']}"
        # cv.putText(image, pose_label, (start_point[0], start_point[1] - 30), cv.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), line_thickness)

        image=draw_chinese_text(image, pose_label, (50, 50), "./STSONG.TTF", 48, (0, 255, 0))

    return image

def draw_chinese_text(image, text, position, font_path, font_size, color):
    """
    在图像上绘制中文文本
    :param image: OpenCV 图像
    :param text: 要绘制的中文文本
    :param position: 文本绘制位置 (x, y)
    :param font_path: 字体文件路径
    :param font_size: 字体大小
    :param color: 文本颜色 (B, G, R)
    :return: 绘制中文文本后的图像
    """
    # 将 OpenCV 图像转换为 PIL 图像
    image_pil = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")

    # 绘制中文文本
    draw.text(position, text, font=font, fill=color)

    # 将 PIL 图像转换回 OpenCV 图像
    image = cv.cvtColor(np.array(image_pil), cv.COLOR_RGB2BGR)
    return image

# # 使用示例 单张图片
# if __name__ == "__main__":
#     yolov8 = YOLOv8Pose(model_path='yolov8n-pose.pt', device='cpu', conf=0.25, iou=0.7)
#     imgs=['stand.png','walk.png','tiao.png','tiao2.png']
#     img_path = imgs[3]

#     img = cv.imread(img_path)
#     results = yolov8.detect(img_path)

#     results_ = yolov8.get_results_as_dicts(results)
#     # print(results_)

#     img_with_results = draw_results(img, results_)

#     # 计算高度以保持宽度为固定像素，高度自适应
#     width = 600
#     height = int(img_with_results.shape[0] * (width / img_with_results.shape[1]))

#     # 检查调整后的高度是否超过900
#     if height > 900:
#         height = 900
#         width = int(img_with_results.shape[1] * (height / img_with_results.shape[0]))

#     # 设置窗口大小
#     cv.namedWindow('img', cv.WINDOW_NORMAL)
#     cv.resizeWindow('img', width, height)
#     cv.imshow('img', img_with_results)
#     cv.waitKey(0)
#     cv.destroyAllWindows()

#     # 直接绘制结果
#     # yolov8.show_results(results, img_path)

if __name__ == "__main__":
    yolov8 = YOLOv8Pose(model_path='yolov8n-pose.pt', device='cpu', conf=0.25, iou=0.7)
    imgs = ['images/stand.png', 'images/walk.png', 'images/tiao.png']  # 图片路径列表，这里假设有三张图片
    processed_images = []                         # 存储处理后的图片结果

    # 处理每张图片并存储结果
    for img_path in imgs:
        img = cv.imread(img_path)
        results = yolov8.detect(img_path)
        results_ = yolov8.get_results_as_dicts(results)
        img_with_results = draw_results(img, results_)
        processed_images.append(img_with_results)

    # 计算每张图片的显示尺寸
    max_width = 600
    for idx, img_result in enumerate(processed_images):
        height = int(img_result.shape[0] * (max_width / img_result.shape[1]))
        if height > 900:
            height = 900
            max_width = int(img_result.shape[1] * (height / img_result.shape[0]))
        processed_images[idx] = cv.resize(img_result, (max_width, height))

    # 显示每张图片的处理结果在独立窗口中
    for idx, img_result in enumerate(processed_images):
        cv.imshow(f'Processed Image {idx+1}', img_result)

    cv.waitKey(0)
    cv.destroyAllWindows()
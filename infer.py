import torch
import cv2 as cv
import numpy as np
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from ultralytics.engine.results import Results
import copy

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


# 关键点的连接关系（COCO数据集的例子）
skeleton = [
    (0, 1), (0, 2), (1, 3), (2, 4),(3, 5),(4, 6),              # 头部到身体
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),                   # 身体到手臂
    (5, 11), (6, 12), (11, 12),                                # 躯干
    (11, 13), (12, 14), (13, 15), (14, 16)                     # 躯干到腿
]


# 推理结果解析
def get_results_as_dicts(results):

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
                    # 遍历每个关键点
                    for point in kpt:
                        x, y, conf = point.tolist()  # 解构关键点信息
                        keypoints.append({
                            "x": x,
                            "y": y,
                            "confidence": conf
                        })

            # 创建结果字典
            result_dict = {
                "class_name": class_name,       # 假设names包含了类别信息
                "boxes": bbox,
                "confidence": confidence,
                "keypoints": keypoints
            }
            results_list.append(result_dict)

    return results_list

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
        keypoints = [(int(kpt['x']), int(kpt['y']), kpt['confidence']) for kpt in res['keypoints']]
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

    return image


# 使用示例
if __name__ == "__main__":
    yolov8 = YOLOv8Pose(model_path='yolov8n-pose.pt', device='cpu', conf=0.25, iou=0.7)
    img_path = '1.png'

    img = cv.imread(img_path)
    results = yolov8.detect(img_path)

    results_=get_results_as_dicts(results)
    print(results_)

    # opencv 绘制结果
    img_with_results = draw_results(img, results_)

    # 计算高度以保持宽度为固定像素，高度自适应
    width=600
    height = int(img_with_results.shape[0] * (width / img_with_results.shape[1]))

    # 检查调整后的高度是否超过900
    if height > 900:
        height = 900
        width = int(img_with_results.shape[1] * (height / img_with_results.shape[0]))

    # 设置窗口大小
    cv.namedWindow('img', cv.WINDOW_NORMAL)
    cv.resizeWindow('img', width, height)
    cv.imshow('img',img_with_results)
    cv.waitKey(0)
    cv.destroyAllWindows()


    # 直接绘制结果
    # yolov8.show_results(results, img_path)

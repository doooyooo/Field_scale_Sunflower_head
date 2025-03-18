from ultralytics import YOLO
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

if __name__ == '__main__':
    # 训练
    # model = YOLO(r"ultralytics\cfg\models\v10\yolov10n.yaml")
    # model = YOLO(r"ultralytics\cfg\models\v10\Dual_Brach_YOLOV10.yaml")
    # model.train(data=r"ultralytics\cfg\datasets\sundata.yaml")


    # model = YOLO(r"ultralytics\cfg\models\v10\yolov10n.yaml")
    # # model.train(data=r"ultralytics\cfg\datasets\filling.yaml")
    # model.train(data=r"ultralytics\cfg\datasets\ripening.yaml")

    # 验证
    model = YOLO(r"runs/detect/train/weights/last.pt")
    model.val(data=r"ultralytics/cfg/datasets/sundata.yaml",batch=1)

    # # 检测
    # model = YOLO(r"runs/detect/train/weights/best.pt")
    # model.predict(source=r"..\images", save=True)

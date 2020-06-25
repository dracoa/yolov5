import mss
from detect_utils import *


def detect():
    weights, imgsz = 'weights/yolov5s.pt', 640
    agnostic_nms, augment = False, False
    conf_thres, iou_thres = 0.4, 0.5
    classes = None

    # Initialize
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    names = model.names if hasattr(model, 'names') else model.modules.names

    # Part of the screen to capture
    monitor = {"top": 40, "left": 0, "width": 800, "height": 640}
    with mss.mss() as sct:
        sc = sct.grab(monitor)  # BGRA
        im0s = np.array(sc)[:, :, :3]  # convert to BGR
        img = letterbox(im0s, new_shape=imgsz)[0]  # resize image (keep ratio)
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=augment)[0]
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)

        im0s = cv2.cvtColor(im0s, cv2.COLOR_BGRA2BGR)  # using cv2 to convert, so that the cv2.rectangle will not complaint
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                for *xyxy, conf, cls in det:
                    print(xyxy)
                    label = '%s %.2f' % (names[int(cls)], conf)
                    plot_one_box(xyxy, im0s, label=label, color=int(cls), line_thickness=3)

        # Show result
        cv2.imshow('My Image', im0s)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    with torch.no_grad():
        detect()

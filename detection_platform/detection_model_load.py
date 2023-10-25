from models.common import DetectMultiBackend
from utils.torch_utils import select_device


def model_load(weights="",  # model.pt path(s)
               device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
               half=False,  # use FP16 half-precision inference
               dnn=False,  # use OpenCV DNN for ONNX inference
               ):
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()
    print("Detection model deployed!")
    return model

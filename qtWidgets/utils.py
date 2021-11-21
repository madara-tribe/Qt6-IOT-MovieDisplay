import sys, os
import cv2
import numpy as np
import torch
from torchvision.utils import save_image, make_grid
import torchvision.transforms as transforms

transforms_ = [
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
transform = transforms.Compose(transforms_)
    
def onnx_predict(x, ort_session):
    inputs = transform(x)
    inputs = np.expand_dims(inputs.to('cpu').detach().numpy().copy(), 0)
    output = ort_session.run(None, {"input1": inputs.astype(np.float32)})[0]
    output_ = post_process(output, onnx_is=True)
    return np.hstack([x, output_])
        
def torch_predict(x, model):
    frame = torch.from_numpy(np.expand_dims((x/127.5)-1, axis=0)).permute((0, 3, 1, 2))
    frame = model(frame.to("cpu").float())
    frame_ = post_process(frame, onnx_is=None)
    return np.hstack([x, frame_])

def post_process(frame, onnx_is=None):
    if onnx_is:
        frame = frame * 127.5 + 127.5
        frame = np.squeeze(frame.transpose(0, 2, 3, 1), axis=0)
        cv2.imwrite('/tmp/pred.png', frame.astype(np.uint8))
        return cv2.imread('/tmp/pred.png')
    else:
        save_image(make_grid(frame, nrow=5, normalize=True), '/tmp/pred.png', normalize=False)
        return cv2.cvtColor(cv2.imread('/tmp/pred.png'), cv2.COLOR_BGR2RGB)

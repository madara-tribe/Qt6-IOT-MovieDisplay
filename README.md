# Versions
```bash
・ cuda 10.1
・ nvidia driver 430.64
・ Python 3.7.11
・ pytorch 1.7.1+cu101
・ torchvision 0.8.2+cu101
・ onnx 1.10.2
・ onnxruntime 1.9.0
・ PySide6 6.2.0
```

# abstract about cycleGAN
cycle GAN mainly training horse to zebra and zebra to horse.
<img src="https://user-images.githubusercontent.com/48679574/139290811-74376ccc-6e25-42a8-9cf9-834dc9fdfda0.png" width="650px">


## pytorch code for speed up
```python
# at data loder
num_workers = 8 if os.cpu_count() > 8 else os.cpu_count()
pin_memory = True
DataLoader(〜〜, num_workers=num_workers, pin_memory=pin_memory)
# before training
torch.backends.cudnn.benchmark =True
```

# Result
## 1. cycleGAN with pyside ML GUI app
```pythonn
$ python3 Qtapp.py
```
![2zebra](https://user-images.githubusercontent.com/48679574/139296110-7c2c3c5f-8937-43b8-a559-e3f8bbf1cbe6.gif)



## 2. result : convrting hourse to zebra
training images is 1139 test image is 150
<b>at 199 epoch</b>

<img src="https://user-images.githubusercontent.com/48679574/139290914-1e7597dd-d408-4945-887d-cc2ccc772b88.png" width="1600px">


## 3. Result with ONNX format
<b>hourse to zebra</b>

```Inference Latency (milliseconds) is 45.3539218902588 [ms]```

<img src="https://user-images.githubusercontent.com/48679574/140451655-a702285e-e886-4bdc-9749-2ac4f13d4ab7.jpg" width="300px"><img src="https://user-images.githubusercontent.com/48679574/140451671-5ef18ba6-2a9c-4cfd-bc5c-677f3fa93a70.png" width="300px">


## 4. Generator and Discriminator loss curve

<img src="https://user-images.githubusercontent.com/48679574/139297260-ae4b1291-9538-4874-8b0a-8e5f6ace3baa.png" width="400px"><img src="https://user-images.githubusercontent.com/48679574/139297265-17c81eaf-112a-4c25-aeb1-c37487cdf070.png" width="400px">


# References
- [Tensorflow GPU, CUDA, CuDNNのバージョン早見表](https://qiita.com/chin_self_driving_car/items/f00af2dbd022b65c9068)
- [PyTorchでの学習・推論を高速化するコツ集](https://qiita.com/sugulu_Ogawa_ISID/items/62f5f7adee083d96a587)

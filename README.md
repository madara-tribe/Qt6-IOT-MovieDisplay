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
<img src="https://user-images.githubusercontent.com/48679574/142752809-9243c8bd-e0bb-4d5d-9798-4a9f4181c85f.png" width="650px">





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

<img src="https://user-images.githubusercontent.com/48679574/142752812-2606162d-2cdb-419b-b6e0-b2d07def95f0.jpg" width="300px"><img src="https://user-images.githubusercontent.com/48679574/142752813-9d69f009-a598-4f1b-8bac-efe908bc392e.png" width="300px">


## 4. Generator and Discriminator loss curve

<img src="https://user-images.githubusercontent.com/48679574/142752865-7a962b27-5c90-4d62-a44c-d36d3328e9b9.png" width="400px"><img src="https://user-images.githubusercontent.com/48679574/142752867-4d6a39bd-b919-4bdb-8ece-e5b1b12ea639.png" width="400px">


# References
- [Tensorflow GPU, CUDA, CuDNNのバージョン早見表](https://qiita.com/chin_self_driving_car/items/f00af2dbd022b65c9068)
- [PyTorchでの学習・推論を高速化するコツ集](https://qiita.com/sugulu_Ogawa_ISID/items/62f5f7adee083d96a587)

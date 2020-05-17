import os
import time

architectures = ["unet", "linknet"]
backbones = ["resnet101", "se_resnet101", "se_resnext101_32x4d"]

for arch in architectures:
    for back in backbones:
        os.system(f"python main.py train -a {arch} -b {back}")
        time.sleep(60)


for model in os.listdir("weights"):
    os.system(
        f"python main.py predict data/sample/images/e1d3e6f6ba_B4DE0FB544INSPIRE-ortho.tif {os.path.join('weights', model)}"
    )

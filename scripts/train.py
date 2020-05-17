import os
import time

architectures = ["unet", "linknet"]
backbones = ["resnet101", "se_resnet101", "se_resnext101_32x4d"]

for arch in architectures:
    for back in backbones:
        os.system(f"python main.py train -a {arch} -b {back}")
        time.sleep(60 * 5)


for model in os.listdir("weights"):
    os.system(
        f"python main.py predict data/sample/images/ec09336a6f_06BA0AF311OPENPIPELINE-ortho.tif {os.path.join('weights', model)}"
    )
    time.sleep(5)

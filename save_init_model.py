import os
import torch
import torchvision

TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models", "test")

def deeplabmobilenetv3(dir=None, num_classes=21):
    if dir is None:
        dir = TEST_DIR

    if not os.path.exists(dir): 
        os.makedirs(dir)

    model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)

    # Custom number of classes (default is 21 for COCO dataset)
    if num_classes != 21:
        model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)

    torch.save(model.state_dict(), os.path.join(dir, "mobilenetv3_test.pth"))

def deeplabresnet101(dir=None, num_classes=21):
    if dir is None:
        dir = TEST_DIR

    if not os.path.exists(dir):
        os.makedirs(dir)

    model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)

    # Custom number of classes (default is 21 for COCO dataset)
    if num_classes != 21:
        model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)

    torch.save(model.state_dict(), os.path.join(dir, "deeplabv3_test.pth"))

if __name__ == "__main__":
    # deeplabmobilenetv3()
    deeplabresnet101()
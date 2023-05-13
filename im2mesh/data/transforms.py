from torchvision import transforms

class ResizeImage(object):
    def __init__(self, img_size):
        if img_size is None or img_size < 1:
            self.transform = transforms.Compose([
                transforms.ToTensor()])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor()])

    def __call__(self, img):
        img = self.transform(img)
        return img
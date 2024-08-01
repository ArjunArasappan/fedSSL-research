from torchvision import transforms


class SimCLRTransform:
    def __init__(self, size=32):
        color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        self.size = size
 
        self.base_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor()]
        )
        
        self.test_transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor()
        ])
    
    def __call__(self, x, augmentData):
        if not augmentData:
            return self.test_transform(x)
        
        return (self.base_transform(x), self.base_transform(x))
                
        


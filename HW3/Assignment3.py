import numpy as np
import torch
import torchvision
import cv2 as cv
import matplotlib.pyplot as plt


class Assignment3:
    def __init__(self) -> None:
        pass

    def torch_image_conversion(self, torch_img):
        img = cv.cvtColor(torch_img, cv.COLOR_BGR2RGB)
        # Convert the image to a PyTorch tensor
        torch_img = torch.from_numpy(img).to(torch.float32)
        '''
        print("Torch image shape:", torch_img.shape)
        print("Torch image dtype:", torch_img.dtype)
        print("Torch Image pixel dtype:", torch_img[3,2,1].dtype)
        '''
        return torch_img

    def brighten(self, torch_img):
        bright_img = torch_img + 100
        '''
        cv.imshow("Brightened Image", bright_img.numpy().astype(np.uint8))
        cv.waitKey(0)
        cv.destroyAllWindows()
        '''
        return bright_img

    def saturation_arithmetic(self, img):
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        torch_img = torch.from_numpy(img_rgb).to(torch.uint8)
        # first convert to int32 to prevent from overflow
        bright_img = torch_img.to(torch.int32) + 100
        # shouldn't used clamp, the expected behavior is uint8 overflow (mod 256)
        # bright_img[bright_img > 255] = 255
        saturated_img = bright_img.to(torch.uint8)
        '''
        cv.imshow("Saturated Image", saturated_img.numpy())
        cv.waitKey(0)
        cv.destroyAllWindows()
        '''
        return saturated_img

    def add_noise(self, torch_img):
        # noise with mean= 0 and  σ=100.0 gray levels
        noise = torch.randn_like(torch_img) * 100.0
        noisy_img = torch_img + noise
        noisy_img /= 255.0
        '''
        # opencv cant show float32 img, so I convert it to uint8
        cv.imshow("Noisy Image", noisy_img.numpy().astype(np.uint8))
        cv.waitKey(0)
        cv.destroyAllWindows()
        '''
        return noisy_img

    def normalization_image(self, img):
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img_float = torch.from_numpy(img_rgb).to(torch.float64)
        mean = img_float.mean(dim=(0, 1))  # shape = [3]
        std = img_float.std(dim=(0, 1))    # shape = [3]
        image_norm = (img_float - mean) / std
        # image_norm = torch.clamp(image_norm, 0.0, 1.0)
        '''
        print("Normalized mean:", image_norm.mean(dim=(0,1)))
        print("Normalized std:", image_norm.std(dim=(0,1)))
        '''
        return image_norm

    def Imagenet_norm(self, img):
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img_float = torch.from_numpy(img_rgb).to(torch.float64)
        img_float = img_float / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float64)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float64)
        ImageNet_norm = (img_float - mean) / std
        ImageNet_norm = torch.clamp(ImageNet_norm, 0.0, 1.0)
        '''
        print("ImageNet normalized mean:", ImageNet_norm.mean(dim=(0,1)))
        print("ImageNet normalized std:", ImageNet_norm.std(dim=(0,1)))
        '''
        return ImageNet_norm

    def dimension_rearrange(self, img):
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img_float = torch.from_numpy(img_rgb).to(torch.float32)
        rearrange = img_float.permute(2, 0, 1).unsqueeze(0)
        '''
        # torch.Size([1, 3, 321, 433])
        print("Shape:", rearrange.shape)
        '''
        return rearrange
    '''
    # left empty
    def chain_rule(self, x, y, z):
        
        return df_dx, df_dy, df_dz, df_dq
    # left empty
    def relu(self, x, w):
        
        return dx, dw
    '''
    # missing stride convolution(cant find the template...)
    def stride(self, img):
        img = torch.from_numpy(img).to(torch.float32)

        # 3x3 Scharr_x filter (from lec 2)
        scharr_x = torch.tensor([[ 3.0,  0.0,  -3.0],
                                [10.0,  0.0, -10.0],
                                [ 3.0,  0.0,  -3.0]], dtype=torch.float32)

        # 0 padding
        img_padded = torch.nn.functional.pad(img.unsqueeze(0).unsqueeze(0),
                                            (1, 1, 1, 1),  # (left, right, top, bottom)
                                            mode='constant', value=0)

        # stride = 2
        H, W = img_padded.shape[2], img_padded.shape[3]
        out_H = (H - 3) // 2 + 1
        out_W = (W - 3) // 2 + 1
        output = torch.zeros((out_H, out_W), dtype=torch.float32)

        for i in range(out_H):
            for j in range(out_W):
                region = img_padded[0, 0, i*2:i*2+3, j*2:j*2+3]
                output[i, j] = torch.sum(region * scharr_x)
        '''
        print("stride output shape:", output.shape)
        print("dtype:", output.dtype)
        print("sample:", output[0, 0])
        '''
        return output


if __name__ == "__main__":
    img = cv.imread("original_image.png")
    assign = Assignment3()
    torch_img = assign.torch_image_conversion(img)
    bright_img = assign.brighten(torch_img)
    saturated_img = assign.saturation_arithmetic(img)
    noisy_img = assign.add_noise(torch_img)
    image_norm = assign.normalization_image(img)
    ImageNet_norm = assign.Imagenet_norm(img)
    rearrange = assign.dimension_rearrange(img)
    cat_img = cv.imread("cat_eye.jpg")
    gray = cv.cvtColor(cat_img, cv.COLOR_BGR2GRAY)
    stride = assign.stride(gray)
    # df_dx, df_dy, df_dz, df_dq = assign.chain_rule(x=-2.0, y=5.0, z=-4.0)
    # dx, dw = assign.relu(x=[-1.0, 2.0], w=[2.0, -3.0, -3.0])

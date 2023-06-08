import cv2
import numpy as np
import torch
import lpips
from skimage.metrics import structural_similarity as ssim
import warnings

warnings.filterwarnings("ignore")

# 載入預訓練的LPIPS模型
loss_fn = lpips.LPIPS(net='alex')

# 自己的手寫字圖片路徑
gt = cv2.imread("./Circle/J/5a9b.png", cv2.IMREAD_GRAYSCALE)
gt = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0).float() / 255.0

# 別人的手寫字圖片路徑
gradient = cv2.imread("./GenYoMin/J/5a9b.png", cv2.IMREAD_GRAYSCALE)
gradient = torch.from_numpy(gradient).unsqueeze(0).unsqueeze(0).float() / 255.0

# 計算MSE
mse = np.mean((gt.squeeze().numpy() - gradient.squeeze().numpy()) ** 2)

# 計算SSIM相似度
ssim_score = ssim(gt.squeeze().numpy(), gradient.squeeze().numpy(), win_size=7)

# 計算LPIPS距離
lpips_distance = loss_fn(gt, gradient)

# Print the score
print("MSE score:", "{:.5f}".format(mse) )
print("SSIM score:", "{:.5f}".format(ssim_score) )
print("LPIPS distance:", "{:.5f}".format(lpips_distance.item()))

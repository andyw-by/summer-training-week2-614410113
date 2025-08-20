import os
import glob
import pandas as pd
from PIL import Image
import torch

class RFMiDDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        # 將第一列（圖片名稱）轉成字串，避免 FutureWarning
        #self.df.iloc[:, 0] = self.df.iloc[:, 0].apply(str)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx, 0]
        label = int(self.df.iloc[idx, 1])

        # 嘗試匹配 .png 檔案
        pattern = os.path.join(self.img_dir, f"{img_id}.png")
        matches = glob.glob(pattern)

        if len(matches) == 0:
            raise FileNotFoundError(f"找不到圖像檔案匹配 {pattern}")

        image_path = matches[0]  # 取第一個匹配檔案
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

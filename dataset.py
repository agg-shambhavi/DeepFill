import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2


class CelebA(Dataset):
    def __init__(self, rootDir):
        self.rootDir = rootDir
        self.images = []
        for filename in os.listdir(self.rootDir):
            if filename.endswith(".jpg"):
                self.images.append(filename)

        self.images_len = len(self.images)

    def __len__(self):
        return self.images_len

    def __getitem__(self, index):
        # load image
        img = self.images[index % self.images_len]
        img_path = os.path.join(self.rootDir, img)
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))

        # load mask
        mask = self.random_ff_mask(256)

        # convert to torch
        img = (
            torch.from_numpy(img.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .contiguous()
        )
        mask = torch.from_numpy(mask.astype(np.float32)).contiguous()

        return (img, mask)

    def random_ff_mask(
        self, mask_size, max_vertex=30, max_length=40, max_angle=4, max_brush_width=10
    ):
        mask = np.zeros((mask_size, mask_size), np.float32)
        numVertex = np.random.randint(max_vertex)
        for i in range(numVertex):
            start_x = np.random.randint(mask_size)
            start_y = np.random.randint(mask_size)
            for j in range(1 + np.random.randint(5)):
                angle = 0.01 + np.random.randint(max_angle)
                if i % 2 == 0:
                    angle = 2 * 3.141 - angle
                length = 10 + np.random.randint(max_length)
                brush_width = 5 + np.random.randint(max_brush_width)
                end_x = (start_x + length * np.sin(angle)).astype(np.int32)
                end_y = (start_y + length * np.cos(angle)).astype(np.int32)
                cv2.line(mask, (start_x, start_y), (end_x, end_y), 1.0, brush_width)
                start_x, start_y = end_x, end_y
        return mask


if __name__ == "__main__":
    root_dir = r"G:\DeepFill\Data\data256x256"
    data = CelebA(root_dir)
    dataloader = DataLoader(data, batch_size=1, shuffle=True)
    img, mask = next(iter(dataloader))
    # print(img.shape)
    # print(mask.shape)
    cv2.imshow(
        "image", cv2.cvtColor(img[0].permute(1, 2, 0).numpy(), cv2.COLOR_BGR2RGB)
    )
    cv2.imshow("MASK tensor", mask[0].numpy())
    input_image = img - mask
    print(input_image.shape)
    cv2.imshow(
        "input image",
        cv2.cvtColor(input_image[0].permute(1, 2, 0).numpy(), cv2.COLOR_BGR2RGB),
    )
    cv2.waitKey()
    cv2.destroyAllWindows()

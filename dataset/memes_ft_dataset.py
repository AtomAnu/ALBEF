import os
import json
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from dataset.utils import pre_question
import numpy as np

class memes_ft_dataset(Dataset):

    def __init__(self, ann_file, transform, img_root, split='train', max_words=50):

        self.split = split
        self.ann = []
        for f in ann_file:
            with open(f, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    self.ann.append(json.loads(line))

        self.transform = transform
        self.img_root = img_root
        self.max_words = max_words

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]

        image_path = os.path.join(self.img_root, ann['img'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        text = ann['text']
        label = ann['label']

        return image, text, torch.tensor(label)
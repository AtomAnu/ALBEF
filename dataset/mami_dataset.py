import os
import json
import random
from PIL import Image
from torch.utils.data import Dataset
from dataset.utils import pre_question
import numpy as np

class mami_dataset(Dataset):

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

        # eos

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]

        image_path = os.path.join(self.img_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        text = ann['text']
        labels = [ann['misogynous'], ann['shaming'], ann['stereotype'], ann['objectification'], ann['violence']]

        if np.sum(labels) > 0:
            labels += [0]
        else:
            labels += [1]

        return image, text, labels
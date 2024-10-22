import os
import numpy as np
import torch
import torchvision.transforms.functional as F

import random
from typing import Tuple

from torch.nn.functional import conv2d
from torchvision import transforms as T
from PIL import Image
from torchvision.transforms import InterpolationMode

from datasets import load_dataset
from util.datasets import create_link_image

class MAEDataset:
    def __init__(
            self,
            image_folder:str=None,
            dataset_name_train:str = None,
            do_train:bool = True,
            do_eval:bool = False,
            max_train_samples = None,
            max_val_samples = None,
            num_proc:int = 8,
            streaming:bool = False,
            batch_size:int = 1000,
            mask_config:float = 0.75,
            img_size:tuple = (224, 224),
            mask_min:float = 0.25,
            mask_max:float = 0.3,
            cache_dir:str = None,
            mean_dataset:Tuple[float, ...] = None,
            std_dataset:Tuple[float, ...] = None,
    ):
        self.image_folder = image_folder
        self.do_train = do_train
        self.do_eval = do_eval
        self.dataset_name_train = dataset_name_train
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples

        self.num_proc = num_proc if num_proc > os.cpu_count() else os.cpu_count()
        self.streaming = streaming
        self.batch_size = batch_size
        self.mask_config = mask_config
        self.img_size = img_size
        self.mask_min = mask_min
        self.mask_max = mask_max
        
        self.cache_dir = cache_dir
        self.link_image = create_link_image(image_folder)
        
        self.mean_dataset = mean_dataset
        self.std_dataset = std_dataset

    def process(self):
        '''
            Process the dataset for training and validation
        '''
        final = {
            'train': None,
            'validation': None
        }
        if self.do_train:
            data = self.load_data(mode='train')
            if self.max_train_samples > 0 and not self.streaming:
                data = data.select(range(self.max_train_samples))
            final['train'] = self.prossersor(data)

        if self.do_eval:
            data = self.load_data(mode='validation')
            if self.max_val_samples > 0:
                data = data.select(range(self.max_val_samples)) 
            final['validation'] = self.prossersor(data)
            
        return final

    def load_data(self, mode='train'):
        '''
            Create a dataset from Hugging Face
        '''
        if mode == "validation" or not self.streaming:
            ds = load_dataset(
                self.dataset_name_train,
                num_proc=self.num_proc,
                cache_dir=self.cache_dir,
            )[mode]
            ds = ds.shuffle()
        else:
            ds = load_dataset(
                self.dataset_name_train,
                streaming=True,
                cache_dir=self.cache_dir,
                desc='Download image: '
            )[mode]
            
        ds = ds.filter(
                self.filter_fn,
                batched=True, 
                num_proc=self.num_proc,
                desc=f'Filter mask ratio {self.mask_min} - {self.mask_max}'
        )

        return ds

    def prossersor(self, dataset):
        '''
            Preprocess the dataset for training and validation
        '''
        dataset = dataset.with_transform(self.preprocess_function)
        return dataset

    def preprocess_function(self, examples):
        """
            Preprocess a batch of images by applying transforms
        """
        images, masks = [], []
        for image, mask in zip(examples["image"], examples["mask"]):
            image = self.link_image[image]
            image, mask = self.transform_input(image, mask)
            images.append(image)
            masks.append(mask)
        pixel_values = torch.stack(images)
        pixel_values_mask = torch.stack(masks)
        return {
            'pixel_values':pixel_values, 
            'pixel_values_mask':pixel_values_mask
        }

    def filter_fn(self, examples):
        '''
            Filter function to remove mask < 0.25
        '''
        transform_mask = T.Compose([lambda image: image.convert("1"), T.Resize((224, 224)), T.ToTensor()])
        mask = torch.stack([transform_mask(each) for each in examples['mask']])
        kernel = torch.ones(1, 1, 16, 16)
        output_tensors = conv2d(mask, kernel, stride=16, padding=0)
        output_tensors = (output_tensors > 0).float()
        area_min = output_tensors.size(2) * output_tensors.size(3) * self.mask_min
        area_max = output_tensors.size(2) * output_tensors.size(3) * self.mask_max
        return (torch.sum(output_tensors, dim=(1, 2, 3)) > area_min) & (torch.sum(output_tensors, dim=(1, 2, 3)) < area_max)
    
    def transform_input(self, image_path, mask, mode='train'):
        '''
            Apply transforms to the image and mask
        '''
        image = Image.open(image_path)
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        if mask.mode != "1":
            mask = mask.convert("1")
            
        transform_object = T.Compose([
            T.RandomResizedCrop(self.img_size, scale=(0.2, 1.0), interpolation=InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ])
        transform_color = T.Compose([
            T.Normalize(mean=self.mean_dataset, std=self.std_dataset)
        ])
        image = transform_color(transform_object(image))
        mask = transform_object(mask)
        return image, mask
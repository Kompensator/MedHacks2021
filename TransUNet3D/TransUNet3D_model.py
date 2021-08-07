import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import os
from unet_parts import *
from matplotlib import pyplot as plt
import math
from config import get_config


class Attention(nn.Module):
    """ Multi head self attention to encode 2D image
        - Embeds h x w image into n 16x16 patches by convolution
        - Embeds position info by tensor addition
        - Applies multihead attention + an extra output filter
            for good measure
        - Applies 2x linear layers to either decode to image
            OR continue to next attention layer
    """
    def __init__(self, config, in_channels, img_size, output_image=True):
        super(Attention, self).__init__()
        self.output_image = output_image
        self.patch_size = 2            # NOTE higher patch_size conv doesn't work with 4 down conv
        self.n_patch = int((img_size/self.patch_size) ** 3)          # NOTE assuming a cube shaped embedding
        self.patch_embeddings = nn.Conv3d(in_channels=in_channels, 
                                            out_channels=config.embedding_size,
                                            kernel_size=(self.patch_size, self.patch_size, self.patch_size),
                                            stride=(self.patch_size, self.patch_size, self.patch_size))
        self.positional_embeddings = nn.Parameter(torch.zeros(1, self.n_patch, config.embedding_size))
        self.embedding_dropout = nn.Dropout(0.2)

        self.attention_norm = nn.LayerNorm(config.embedding_size)

        self.num_heads = config.num_heads
        self.head_size = int(config.embedding_size / self.num_heads)
        self.all_head_size = self.num_heads * self.head_size
        self.query = nn.Linear(config.embedding_size, self.all_head_size)
        self.key = nn.Linear(config.embedding_size,  self.all_head_size)
        self.value = nn.Linear(config.embedding_size, self.all_head_size)

        self.out = nn.Linear(config.embedding_size, config.embedding_size)
        self.attention_dropout = nn.Dropout(0.2)
        self.projection_dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=-1)

        self.mlp_norm = nn.LayerNorm(config.embedding_size)
        self.mlp_act = nn.GELU()
        self.mlp_dropout = nn.Dropout(0.2)
        mlp_channels = config.mlp_size         # seems arbitrary
        self.mlp_fc1 = nn.Linear(config.embedding_size, mlp_channels)
        if self.output_image:           # NOTE in TransUNet V1 they use another conv before bilinear upsampling
            self.mlp_conv_transpose = nn.ConvTranspose2d(in_channels=mlp_channels,
                                                        out_channels=in_channels,
                                                        kernel_size=(self.patch_size, self.patch_size),
                                                        stride=(self.patch_size, self.patch_size))
        else:                                                           
            self.mlp_fc2 = nn.Linear(mlp_channels, config.embedding_size)
        self._init_weights()


    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def _init_weights(self):
        nn.init.xavier_uniform_(self.mlp_fc1.weight)
        nn.init.normal_(self.mlp_fc1.bias, std=1e-6)
        if not self.output_image:
            nn.init.xavier_uniform_(self.mlp_fc2.weight)
            nn.init.normal_(self.mlp_fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.patch_embeddings(x)
        x = x.flatten(2)        
        x = x.transpose(-1, -2)   # (batch_size, n_patch, embedding_size)
        x = x + self.positional_embeddings
        x = self.embedding_dropout(x)

        h = x
        x = self.attention_norm(x)
        mixed_query = self.query(x)
        mixed_key = self.key(x)
        mixed_value = self.value(x)
        
        Q = self.transpose_for_scores(mixed_query)
        K = self.transpose_for_scores(mixed_key)
        V = self.transpose_for_scores(mixed_value)

        attention = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_size)
        attention_probs = self.softmax(attention)

        context = torch.matmul(attention_probs, V).permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context.size()[:-2] + (self.head_size,)
        context = context.view(*new_context_layer_shape)                # FIXME

        attention_output = self.out(context)
        attention_output = self.projection_dropout(attention_output)

        visualize = False
        if visualize:
            attn_map = attention_probs
        else:
            attn_map = None
        x = attention_output + h

        h = x
        x = self.mlp_norm(x)
        x = self.mlp_fc1(x)
        x = self.mlp_act(x)
        x = self.mlp_dropout(x)
        x = self.mlp_fc2(x)
        x = self.mlp_dropout(x)
        x = x + h
        return x, attn_map


class TransUNetV2(nn.Module):
    """ traditional 2D UNet with attention layer in between
        down sampling operations"""
    def __init__(self, config, in_channels=1, n_classes=6, img_size=256):
        super(TransUNetV2, self).__init__()
        self.config = config
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)

        bilinear = True
        factor = 2
        self.down4 = Down(256, 512 // factor)
        self.attention = Attention(config, 256, img_size=img_size/16, output_image=True)

        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        triple_channel_input = False
        if x.size()[1] == 1 and triple_channel_input:
            x = x.repeat(1, 3 , 1 , 1)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x5 = self.attention(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


if __name__ == "__main__":
    import sys
    sys.path.append(r"C:\Users\dingyi.zhang\Documents\MedHacks2021")
    os.chdir(r'C:\Users\dingyi.zhang\Documents\MedHacks2021')
    from unet_datasets import *

    GPU = False

    config = get_config()
    model = TransUNetV2(config, in_channels=1, n_classes=2, img_size=256)
    if GPU:
        model = model.cuda()

    train_set = AlphaTau3_train(start=0.0, end=0.005)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=1)
    val_loader = torch.utils.data.DataLoader(train_loader, batch_size=1, shuffle=False, num_workers=1)

    max_epoch = config.epoch
    LR = config.learning_rate
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-5)
    CE = nn.CrossEntropyLoss()

    for epoch in range(max_epoch):
        for i_batch, (image_batch, label_batch) in enumerate(train_loader):
            if GPU:
                image_batch = image_batch.cuda()
                label_batch = label_batch.cuda()
            model.train()
            outputs = model(image_batch)
            loss = CE(outputs, label_batch[:].long())
            loss.backward()
            optimizer.step()
            print("Iteration {}\tEpoch {}\tLoss: {}".format(i_batch, epoch, loss.item()))
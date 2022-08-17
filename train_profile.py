"""
Training of DCGAN network with WGAN loss
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model_co_with_ms import DcganD, DcganG, initialize_weights

import torch.profiler as profiler
import numpy as np
from torchvision.utils import save_image
# Hyperparameters etc
device = "cuda" if torch.cuda.is_available() else "cpu"
batchSize = 64
imageSize = 64
nc = 3
nz = 100
ngf = 64
ndf = 64
n_extra_layers = 0
LEARNING_RATE = 5e-5
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 3
Z_DIM = 100
NUM_EPOCHS = 1
FEATURES_CRITIC = 64
FEATURES_GEN = 64
CRITIC_ITERATIONS = 1
WEIGHT_CLIP = 0.01
experiment = './generate'
# test_dir = r'/data/liaozy/lsun'
test_dir = r'F:\DeepLearningData\vision\imagenet-mini\train'
os.makedirs(experiment, exist_ok=True)

def main():
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA], #
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/wgan')

    ) as prof:
        transform = transforms.Compose(
            [
                transforms.Resize(IMAGE_SIZE),
                transforms.CenterCrop(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
                ),
            ]
        )

        t_begin = time.time()
        # dataset = datasets.MNIST(root="dataset/", transform=transform, download=True)
        dataset = datasets.ImageFolder(test_dir, transform=transform)
        #comment mnist and uncomment below if you want to train on CelebA dataset
        #dataset = datasets.ImageFolder(root="celeb_dataset", transform=transform)
        data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
        length = len(data_loader)
        # initialize gen and disc/critic
        critic = DcganD(imageSize, nz, nc, ndf, n_extra_layers).to(device)
        gen = DcganG(imageSize, nz, nc, ngf, n_extra_layers).to(device)
        initialize_weights(gen)
        initialize_weights(critic)

        # initializate optimizer
        opt_gen = optim.RMSprop(gen.parameters(), lr=LEARNING_RATE)
        opt_critic = optim.RMSprop(critic.parameters(), lr=LEARNING_RATE)


        fixed_noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1).to(device)

        gen.train()
        critic.train()
        gen_iterations = 0
        t0 = time.time()
        for epoch in range(NUM_EPOCHS):
            # Target labels not needed! <3 unsupervised
            for i, (data, _) in enumerate(data_loader):
                real = data.to(device)
                cur_batch_size = real.shape[0]
                # noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1).to(device)

                # Train Critic: max E[critic(real)] - E[critic(fake)]

                noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
                fake = gen(noise)
                critic_real = critic(real).reshape(-1)
                critic_fake = critic(fake).reshape(-1)
                loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake))
                critic.zero_grad()
                loss_critic.backward(retain_graph=True)
                opt_critic.step()

                # clip critic weights between -0.01, 0.01
                for p in critic.parameters():
                    p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

                # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
                gen_fake = critic(fake).reshape(-1)
                loss_gen = -torch.mean(gen_fake)
                gen.zero_grad()
                loss_gen.backward()
                opt_gen.step()
                prof.step()
                t1 = time.time()
                gen_iterations += 1
                if gen_iterations % 50 == 0:
                    print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f'
                          % (epoch, NUM_EPOCHS, i, length, gen_iterations,
                             loss_critic.item(), loss_gen.item()))
                    print('step_cost: %.4f seconds' % (float(t1 - t0)))
                    break

                t0 = t1
            torch.save(gen.state_dict(), '{0}/netG_epoch_{1}.ckpt'.format(experiment, epoch))
            torch.save(critic.state_dict(), '{0}/netD_epoch_{1}.ckpt'.format(experiment, epoch))
        t_end = time.time()
        print('total_cost: %.4f seconds' % (float(t_end - t_begin)))

        print("Train success!")

    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))

if __name__ == '__main__':
    main()

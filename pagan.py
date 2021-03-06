from __future__ import print_function

import argparse
import os
import random
from collections import OrderedDict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torch.nn.utils import spectral_norm
from tqdm import tqdm

from utils import add_channel, compute_fid, compute_metrics

from download import download_celeb_a


class Generator(nn.Module):
    def __init__(self, ngpu, nc, nz, ngf):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.ngf = ngf
        self.fc = nn.Linear(nz, 8*8*ngf*8)
        self.main = nn.Sequential(
            # state size. (ngf*8) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 3, 1, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            raise NotImplementedError
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            x = F.relu(self.fc(input))
            x = x.view(-1, self.ngf*8, 8, 8)
            output = self.main(x)
        return output


class Discriminator(nn.Module):
    def __init__(self, ngpu, nc, ndf):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.ndf = ndf
        self.main = nn.Sequential(
            OrderedDict(
                [
                    # input is (nc) x 64 x 64
                    ("conv1", spectral_norm(nn.Conv2d(nc, ndf, 3, 1, 1, bias=False))),
                    ("lrelu1", nn.LeakyReLU(0.2, inplace=True)),
                    # state size. (ndf) x 64 x 64
                    ("conv2", spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False))),
                    ("lrelu2", nn.LeakyReLU(0.2, inplace=True)),
                    # state size. (ndf*2) x 32 x 32
                    ("conv3", spectral_norm(nn.Conv2d(ndf * 2, ndf * 2, 3, 1, 1, bias=False))),
                    ("lrelu3", nn.LeakyReLU(0.2, inplace=True)),
                    # state size. (ndf*2) x 32 x 32
                    ("conv4", spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False))),
                    ("lrelu4", nn.LeakyReLU(0.2, inplace=True)),
                    # state size. (ndf*4) x 16 x 16
                    ("conv5", spectral_norm(nn.Conv2d(ndf * 4, ndf * 4, 3, 1, 1, bias=False))),
                    ("lrelu5", nn.LeakyReLU(0.2, inplace=True)),
                    # state size. (ndf*4) x 16 x 16
                    ("conv6", spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False))),
                    ("lrelu6", nn.LeakyReLU(0.2, inplace=True)),
                    # state size. (ndf*8) x 8 x 8
                    ("conv7", spectral_norm(nn.Conv2d(ndf * 8, ndf * 8, 3, 1, 1, bias=False))),
                    ("lrelu7", nn.LeakyReLU(0.2, inplace=True)),
                    # state size. (ndf*8) x 8 x 8
                ]
            )
        )
        self.fc = nn.Linear((ndf*8) * 8 * 8, 1)

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            raise NotImplementedError
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            x = self.main(input)
            x = x.view(-1, (self.ndf*8) * 8 * 8)
            output = torch.sigmoid(self.fc(x))

        return output.squeeze()


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def main(opt):
    writer = SummaryWriter(log_dir="logs/pagan/{}/lr={}_beta1={}_al={}_randomSeed={}/".format(opt.dataset, opt.lr, opt.beta1, opt.al, opt.manualSeed))

    if opt.dataset in ["imagenet", "folder", "lfw"]:
        # folder dataset
        dataset = dset.ImageFolder(
            root=opt.dataroot,
            transform=transforms.Compose(
                [
                    transforms.Resize(opt.imageSize),
                    transforms.CenterCrop(opt.imageSize),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )
    elif opt.dataset == "lsun":
        dataset = dset.LSUN(
            root=opt.dataroot,
            classes=["bedroom_train"],
            transform=transforms.Compose(
                [
                    transforms.Resize(opt.imageSize),
                    transforms.CenterCrop(opt.imageSize),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )
    elif opt.dataset == "cifar10":
        dataset = dset.CIFAR10(
            root=opt.dataroot,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Resize(opt.imageSize),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )
    elif opt.dataset == "mnist":
        dataset = dset.MNIST(
            root=opt.dataroot,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Resize(opt.imageSize),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            ),
        )
    elif opt.dataset == "fashionmnist":
        dataset = dset.FashionMNIST(
            root=opt.dataroot,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Resize(opt.imageSize),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            ),
        )
    elif opt.dataset == "celebA":
        download_celeb_a("data")
        dataset = dset.ImageFolder(
            root="data/celebA",
            transform=transforms.Compose(
                [
                    transforms.Resize(opt.imageSize),
                    transforms.CenterCrop(opt.imageSize),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )
    elif opt.dataset == "fake":
        dataset = dset.FakeData(
            image_size=(3, opt.imageSize, opt.imageSize),
            transform=transforms.ToTensor(),
        )
    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True, num_workers=int(opt.workers)
    )

    device = torch.device("cuda:0" if opt.cuda else "cpu")
    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    ndf = int(opt.ndf)
    nc = 1 if opt.dataset in {"mnist", "fashionmnist"} else 3

    netG = Generator(ngpu, nc, nz, ngf).to(device)
    netG.apply(weights_init)
    if opt.netG != "":
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    netD = Discriminator(ngpu, nc, ndf).to(device)
    augmentation_level = opt.al
    netD.main.conv1 = spectral_norm(nn.Conv2d(nc+augmentation_level, ndf, 3, 1, 1, bias=False)).to(device)
    netD.apply(weights_init)
    if opt.netD != "":
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)

    criterion = nn.BCELoss()

    fixed_noise = torch.rand(opt.batch_size, nz, device=device)*2-1

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))

    global_step = 0
    last_augmentation_step = 0
    kid_score_history = []

    for epoch in range(opt.epochs):

        for i, data in enumerate(dataloader, start=0):

            if global_step % opt.augmentation_interval == 0:
                print("Global step: {}. Computing metrics...".format(global_step))
                samples = random.sample(range(len(dataset)), opt.fid_batch)
                real_samples = [dataset[s][0] for s in samples]
                real_samples = torch.stack(real_samples, dim=0).to(device)
                fake_samples = []
                with torch.no_grad():
                    z = torch.rand(opt.fid_batch, nz, device=device)*2-1
                    for k in tqdm(range(opt.fid_batch // opt.batch_size), desc="Generating fake images"):
                        z_ = z[k * opt.batch_size : (k + 1) * opt.batch_size]
                        fake_samples.append(netG(z_))
                    fake_samples = torch.cat(fake_samples, dim=0).to(device)
                print("Computing KID and FID...")                
                kid, fid = compute_metrics(real_samples, fake_samples)
                print("FID: {:.4f}".format(fid))
                writer.add_scalar("metrics/fid", fid, global_step)
                print("KID: {:.4f}".format(kid))
                writer.add_scalar("metrics/kid", kid, global_step)
                if (len(kid_score_history) >= 2
                    and kid >= (kid_score_history[-1] + kid_score_history[-2]) * 19 / 40
                ):  # (last - KID) smaller than 5% of last
                    # TODO decrease generator LR (paper is not clear)
                    augmentation_level += 1
                    last_augmentation_step = global_step
                    netD.main.conv1 = spectral_norm(nn.Conv2d(nc+augmentation_level, ndf, 3, 1, 1, bias=False)).to(device)
                    netD.main.conv1.apply(weights_init)
                    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
                    print("Augmentation level increased to {}".format(augmentation_level))
                    kid_score_history = []
                else:
                    kid_score_history.append(kid)
                
                writer.add_scalar("augmentation_level", augmentation_level, global_step)

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real = data[0].to(device)
            batch_size = real.size(0)
            if augmentation_level > 0:
                p = min(0.5*(global_step-last_augmentation_step)/opt.tr, 0.5)
                if augmentation_level > 1:
                    augmentation_bits_old = np.random.randint(0, 2, size=(batch_size, augmentation_level-1))
                    augmentation_bits_new = np.where(np.random.rand(batch_size, 1) < p, np.ones((batch_size,1)), np.zeros((batch_size,1)))
                    augmentation_bits = np.concatenate((augmentation_bits_old, augmentation_bits_new), axis=1)
                else:
                    augmentation_bits = np.where(np.random.rand(batch_size,1) < p, np.ones((batch_size,1)), np.zeros((batch_size, 1)))
            else:
                augmentation_bits = None

            real_augmented, real_labels_augmented = add_channel(
                real, augmentation_bits, real=True
            )
            output = netD(real_augmented)
            errD_real = criterion(output, real_labels_augmented)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.rand(batch_size, nz, device=device)*2-1
            fake = netG(noise)
            fake_augmented, fake_labels_augmented = add_channel(
                fake, augmentation_bits, real=False
            )

            output = netD(fake_augmented.detach())
            errD_fake = criterion(output, fake_labels_augmented)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            output = netD(fake_augmented)
            errG = criterion(output, 1-fake_labels_augmented) # fake labels are real for generator cost
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            if i % opt.log_interval == 0:
                print(
                    "[{}/{}][{}/{}] Loss_D: {:.4f} Loss_G: {:.4f} D(x): {:.4f} D(G(z)): {:.4f} / {:.4f}".format(
                        epoch,
                        opt.epochs,
                        i,
                        len(dataloader),
                        errD.item(),
                        errG.item(),
                        D_x,
                        D_G_z1,
                        D_G_z2,
                    )
                )
                writer.add_scalar("discriminator/loss", errD.item(), global_step)
                writer.add_scalar("generator/loss", errG.item(), global_step)
                writer.add_scalar("discriminator/mean", D_x, global_step)
                writer.add_scalar("generator/mean1", D_G_z1, global_step)
                writer.add_scalar("generator/mean2", D_G_z2, global_step)

            if i % opt.save_interval == 0 or i == len(dataloader) - 1:
                if global_step == 0:
                    x = vutils.make_grid(
                        real, normalize=True
                    )
                    writer.add_image('Real images', x, global_step)
                x = vutils.make_grid(
                    fake, normalize=True
                )
                writer.add_image('Generated images', x, global_step)
                vutils.save_image(
                    real, "%s/real_%s.png" % (opt.outi, opt.dataset), normalize=True
                )
                fake = netG(fixed_noise)
                vutils.save_image(
                    fake.detach(),
                    "%s/fake_%s_epoch_%03d.png" % (opt.outi, opt.dataset, epoch),
                    normalize=True,
                )
        
            global_step += 1
        # do checkpointing
        torch.save(
            netG.state_dict(),
            "%s/netG_%s_last.pth" % (opt.outc, opt.dataset),
        )
        torch.save(
            netD.state_dict(),
            "%s/netD_%s_last.pth" % (opt.outc, opt.dataset),
        )
        if epoch%20 == 0:
            torch.save(
                netG.state_dict(),
                "%s/netG_%s_epoch_%d.pth" % (opt.outc, opt.dataset, epoch),
            )
            torch.save(
                netD.state_dict(),
                "%s/netD_%s_epoch_%d.pth" % (opt.outc, opt.dataset, epoch),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        dest="dataset", help="cifar10 | lsun | imagenet | folder | lfw | mnist | fake"
    )
    parser.add_argument("--dataroot", default="", help="path to dataset")
    parser.add_argument(
        "--workers", type=int, help="number of data loading workers", default=4
    )
    parser.add_argument("--batch-size", type=int, default=64, help="input batch size")
    parser.add_argument(
        "--imageSize",
        type=int,
        default=64,
        help="the height / width of the input image to network",
    )
    parser.add_argument(
        "--nz", type=int, default=128, help="size of the latent z vector"
    )
    parser.add_argument(
        "--ngf", type=int, default=64, help="number of generator filters"
    )
    parser.add_argument(
        "--ndf", type=int, default=64, help="number of discriminator filters"
    )
    parser.add_argument(
        "--epochs", type=int, default=400, help="number of epochs to train for"
    )
    parser.add_argument(
        "--lr", type=float, default=0.0002, help="learning rate, default=0.0002"
    )
    parser.add_argument(
        "--beta1", type=float, default=0.5, help="beta1 for adam. default=0.5"
    )
    parser.add_argument(
        "--beta2", type=float, default=0.999, help="beta2 for adam. default=0.999"
    )
    parser.add_argument(
        "--tr", type=int, default=5000, help="steps for p to reach 0.5"
    )
    parser.add_argument(
        "--al", type=int, default=0, help="starting augmentation level"
    )
    parser.add_argument("--cpu", action="store_true", help="use cpu")
    parser.add_argument("--ngpu", type=int, default=1, help="number of GPUs to use")
    parser.add_argument(
        "--netG", default="", help="path to netG (to continue training)"
    )
    parser.add_argument(
        "--netD", default="", help="path to netD (to continue training)"
    )
    parser.add_argument(
        "--outi", default="./generated/", help="folder to output images"
    )
    parser.add_argument(
        "--outc", default="./checkpoints/", help="folder to output model checkpoints"
    )
    parser.add_argument("--log-interval", default=25, type=int, help="log interval")
    parser.add_argument(
        "--augmentation-interval", default=10000, help="augmentation interval"
    )
    parser.add_argument(
        "--kid-batch", default=9984, type=int, help="how many images to use to compute kid"
    )
    parser.add_argument(
        "--fid-batch", default=9984, type=int, help="how many images to use to compute fid"
    )
    parser.add_argument("--save-interval", default=100, type=int, help="save interval")
    parser.add_argument("--manualSeed", default=None, type=int, help="manual seed")

    opt = parser.parse_args()
    print(opt)

    if opt.dataroot == "":
        opt.dataroot = "data/{}".format(opt.dataset)

    try:
        os.makedirs(opt.outi)
    except OSError:
        pass

    try:
        os.makedirs(opt.outc)
    except OSError:
        pass

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    cudnn.benchmark = True

    if not torch.cuda.is_available() or opt.cpu:
        opt.cuda = False
    else:
        opt.cuda = True

    main(opt)

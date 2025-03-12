from libs import *
from util import *
from model import *
from trainer import *

with open("ct_to_mri/sycle_gan/config.yaml", "r") as file:
    config = yaml.safe_load(file)

dataset_name        = config["data"]["name"]
channels            = config["data"]["channels"]
img_height          = config["data"]["img_height"]
img_width           = config["data"]["img_width"]
n_residual_blocks   = config["trainer"]["n_residual_blocks"]
lr                  = config["trainer"]["lr"]
b1                  = config["trainer"]["b1"]
b2                  = config["trainer"]["b2"]
n_epochs            = config["trainer"]["n_epochs"]
init_epoch          = config["trainer"]["init_epoch"]
decay_epoch         = config["trainer"]["decay_epoch"]
lambda_cyc          = config["trainer"]["lambda_cyc"]
lambda_id           = config["trainer"]["lambda_id"]
n_cpu               = config["trainer"]["n_cpu"]
batch_size          = config["trainer"]["batch_size"]
sample_interval     = config["trainer"]["sample_interval"]
checkpoint_interval = config["trainer"]["checkpoint_interval"]

transforms_ = [
    transforms.Resize(int(img_height * 1.12), Image.BICUBIC),
    transforms.RandomCrop((img_height, img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

# Create sample and checkpoint directories
os.makedirs(f"sampleimages/{dataset_name}", exist_ok=True)
os.makedirs(f"saved_models/{dataset_name}", exist_ok=True)

# Losses
criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

input_shape = (channels, img_height, img_width)

# Initialize generator and discriminator
G_AB = GeneratorResNet(input_shape, n_residual_blocks)
G_BA = GeneratorResNet(input_shape, n_residual_blocks)
D_A = Discriminator(input_shape)
D_B = Discriminator(input_shape)

# Setup CUDA
cuda = torch.cuda.is_available()
if cuda:
    G_AB = G_AB.cuda()
    G_BA = G_BA.cuda()
    D_A = D_A.cuda()
    D_B = D_B.cuda()
    criterion_GAN.cuda()
    criterion_cycle.cuda()
    criterion_identity.cuda()

# Initialize weights
G_AB.apply(weights_init_normal)
G_BA.apply(weights_init_normal)
D_A.apply(weights_init_normal)
D_B.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=lr, betas=(b1, b2)
)
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=lr/3, betas=(b1, b2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=lr/3, betas=(b1, b2))

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(n_epochs, init_epoch, decay_epoch).step
)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_A, lr_lambda=LambdaLR(n_epochs, init_epoch, decay_epoch).step
)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_B, lr_lambda=LambdaLR(n_epochs, init_epoch, decay_epoch).step
)

# Buffers of previously generated samples
fake_A_buffer = ReplayBuffer(max_size=50)
fake_B_buffer = ReplayBuffer(max_size=50)

# Training data loader
dataloader = DataLoader(
    ImageDataset(f"ct_to_mri/Dataset_processed/{dataset_name}", transforms_=transforms_, unaligned=True),
    batch_size=batch_size,
    shuffle=True,
    num_workers=n_cpu,
)

# Test data loader
val_dataloader = DataLoader(
    ImageDataset(f"ct_to_mri/Dataset_processed/{dataset_name}", transforms_=transforms_, unaligned=True, mode="test"),
    batch_size=5,
    shuffle=True,
    num_workers=1,
)

# Initialize the Trainer
trainer = Trainer(
    G_AB, G_BA, D_A, D_B, dataloader, val_dataloader, fake_A_buffer, fake_B_buffer, 
    criterion_GAN, criterion_cycle, criterion_identity, 
    optimizer_G, optimizer_D_A, optimizer_D_B, 
    lr_scheduler_G, lr_scheduler_D_A, lr_scheduler_D_B, 
    lambda_cyc, lambda_id, sample_interval, n_epochs, dataset_name, batch_size
)

if __name__ == '__main__':
    # Training loop
    for epoch in range(init_epoch, n_epochs):
        loss_G, loss_D = trainer.train(epoch)  # Train the model

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        # Save model checkpoint
        if checkpoint_interval != -1 and epoch % checkpoint_interval == 0:
            # Create checkpoint directory
            os.makedirs(f"saved_models/{dataset_name}", exist_ok=True)
            
            torch.save(G_AB.state_dict(), f"saved_models/{dataset_name}/G_AB_{epoch}.pth")
            torch.save(G_BA.state_dict(), f"saved_models/{dataset_name}/G_BA_{epoch}.pth")
            torch.save(D_A.state_dict(), f"saved_models/{dataset_name}/D_A_{epoch}.pth")
            torch.save(D_B.state_dict(), f"saved_models/{dataset_name}/D_B_{epoch}.pth")


from libs import *
from util import *
from model import *

class Trainer:
    def __init__(self, G_AB, G_BA, D_A, D_B, dataloader, val_dataloader, fake_A_buffer, fake_B_buffer,
                 criterion_GAN, criterion_cycle, criterion_identity, optimizer_G, optimizer_D_A, optimizer_D_B,
                 lr_scheduler_G, lr_scheduler_D_A, lr_scheduler_D_B, 
                 lambda_cyc, lambda_id, sample_interval, n_epochs, dataset_name, batch_size):
        self.G_AB = G_AB
        self.G_BA = G_BA
        self.D_A = D_A
        self.D_B = D_B
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.fake_A_buffer = fake_A_buffer
        self.fake_B_buffer = fake_B_buffer
        self.criterion_GAN = criterion_GAN
        self.criterion_cycle = criterion_cycle
        self.criterion_identity = criterion_identity
        self.optimizer_G = optimizer_G
        self.optimizer_D_A = optimizer_D_A
        self.optimizer_D_B = optimizer_D_B
        self.lr_scheduler_G = lr_scheduler_G
        self.lr_scheduler_D_A = lr_scheduler_D_A
        self.lr_scheduler_D_B = lr_scheduler_D_B
        self.lambda_cyc = lambda_cyc
        self.lambda_id = lambda_id
        self.sample_interval = sample_interval
        self.n_epochs = n_epochs
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.prev_time = time.time()

    def sample_images(self, batches_done):
        """Saves a generated sample from the test set"""
        imgs = next(iter(self.val_dataloader))
        self.G_AB.eval()
        self.G_BA.eval()
        real_A = Variable(imgs["A"].type(Tensor))
        fake_B = self.G_AB(real_A)
        real_B = Variable(imgs["B"].type(Tensor))
        fake_A = self.G_BA(real_B)
        # Arange images along x-axis
        real_A = make_grid(real_A, nrow=5, normalize=True)
        real_B = make_grid(real_B, nrow=5, normalize=True)
        fake_A = make_grid(fake_A, nrow=5, normalize=True)
        fake_B = make_grid(fake_B, nrow=5, normalize=True)
        # Arange images along y-axis
        image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
        save_image(image_grid, f"sampleimages/{self.dataset_name}/{batches_done}.png", normalize=False)

    def train(self, epoch):
        for i, batch in enumerate(self.dataloader):
            # (1) Set model input
            real_A = Variable(batch["A"].type(Tensor))
            real_B = Variable(batch["B"].type(Tensor))

            # (2) Adversarial ground truths
            valid = Variable(Tensor(np.ones((real_A.size(0), *self.D_A.output_shape))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((real_A.size(0), *self.D_A.output_shape))), requires_grad=False)

            # (3) Train Generators
            self.G_AB.train()
            self.G_BA.train()
            self.optimizer_G.zero_grad()

            # (4) Identity loss
            loss_id_A = self.criterion_identity(self.G_BA(real_A), real_A)
            loss_id_B = self.criterion_identity(self.G_AB(real_B), real_B)

            loss_identity = (loss_id_A + loss_id_B) / 2

            # (5) GAN loss
            fake_B = self.G_AB(real_A)
            loss_GAN_AB = self.criterion_GAN(self.D_B(fake_B), valid)
            fake_A = self.G_BA(real_B)
            loss_GAN_BA = self.criterion_GAN(self.D_A(fake_A), valid)

            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            # (6) Cycle loss
            recov_A = self.G_BA(fake_B)
            loss_cycle_A = self.criterion_cycle(recov_A, real_A)
            recov_B = self.G_AB(fake_A)
            loss_cycle_B = self.criterion_cycle(recov_B, real_B)

            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            # (7) Total loss
            loss_G = loss_GAN + self.lambda_cyc * loss_cycle + self.lambda_id * loss_identity

            loss_G.backward()
            self.optimizer_G.step()

            # (8) Train Discriminator A
            self.optimizer_D_A.zero_grad()

            # (9) Real loss
            loss_real = self.criterion_GAN(self.D_A(real_A), valid)
            # (10) Fake loss (on batch of previously generated samples)
            fake_A_ = self.fake_A_buffer.push_and_pop(fake_A)
            loss_fake = self.criterion_GAN(self.D_A(fake_A_.detach()), fake)
            # (11) Total loss
            loss_D_A = (loss_real + loss_fake) / 2

            loss_D_A.backward()
            self.optimizer_D_A.step()

            # (12) Train Discriminator B
            self.optimizer_D_B.zero_grad()

            # (13) Real loss
            loss_real = self.criterion_GAN(self.D_B(real_B), valid)
            # (14) Fake loss (on batch of previously generated samples)
            fake_B_ = self.fake_B_buffer.push_and_pop(fake_B)
            loss_fake = self.criterion_GAN(self.D_B(fake_B_.detach()), fake)
            # (15) Total loss
            loss_D_B = (loss_real + loss_fake) / 2

            loss_D_B.backward()
            self.optimizer_D_B.step()

            loss_D = (loss_D_A + loss_D_B) / 2

            # (16) Print log
            batches_done = epoch * len(self.dataloader) + i
            batches_left = self.n_epochs * len(self.dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - self.prev_time))
            self.prev_time = time.time()

            sys.stdout.write(
                f"\r[Epoch {epoch}/{self.n_epochs}] [Batch {i}/{len(self.dataloader)}] [D loss: {loss_D.item()}] "
                f"[G loss: {loss_G.item()}, adv: {loss_GAN.item()}, cycle: {loss_cycle.item()}, identity: {loss_identity.item()}] "
                f"ETA: {time_left}"
            )

            # (17) If at sample interval save image
            if batches_done % self.sample_interval == 0:
                self.sample_images(batches_done)

            return loss_G, loss_D


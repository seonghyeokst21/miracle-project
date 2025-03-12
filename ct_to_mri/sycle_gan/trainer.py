from libs import *
from util import *
from model import *

# Trainer 클래스에 추가된 부분
class Trainer:
    def __init__(self, G_AB, G_BA, D_A, D_B, dataloader, val_dataloader, fake_A_buffer, fake_B_buffer,
                 criterion_GAN, criterion_cycle, criterion_identity, optimizer_G, optimizer_D_A, optimizer_D_B,
                 lr_scheduler_G, lr_scheduler_D_A, lr_scheduler_D_B, 
                 lambda_cyc, lambda_id, sample_interval, n_epochs, dataset_name, batch_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Device 설정
        self.G_AB = G_AB.to(self.device)
        self.G_BA = G_BA.to(self.device)
        self.D_A = D_A.to(self.device)
        self.D_B = D_B.to(self.device)
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.fake_A_buffer = fake_A_buffer
        self.fake_B_buffer = fake_B_buffer
        self.criterion_GAN = criterion_GAN.to(self.device)
        self.criterion_cycle = criterion_cycle.to(self.device)
        self.criterion_identity = criterion_identity.to(self.device)
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

        # 손실 값들을 기록할 리스트
        self.d_losses = []
        self.g_losses = []
        self.gan_losses = []
        self.cycle_losses = []
        self.identity_losses = []
        self.epochs = []

        self.d_losses_avg = []
        self.g_losses_avg = []
        self.gan_losses_avg = []
        self.cycle_losses_avg = []
        self.identity_losses_avg = []
        self.epochs_avg = []

        self.initial_noise_stddev = 0.35  # 초기 노이즈 강도
        self.final_noise_stddev = 0.0    # 500 에폭 이후에는 노이즈 없음
        self.total_decay_epochs = 500    # 500 에폭 동안 선형적으로 감소

    def sample_images(self, batches_done):
        """Saves a generated sample from the test set"""
        imgs = next(iter(self.val_dataloader))
        self.G_AB.eval()
        self.G_BA.eval()
        real_A = Variable(imgs["A"].type(Tensor)).to(self.device)
        fake_B = self.G_AB(real_A)
        real_B = Variable(imgs["B"].type(Tensor)).to(self.device)
        fake_A = self.G_BA(real_B)
        # Arrange images along x-axis
        real_A = make_grid(real_A, nrow=5, normalize=True)
        real_B = make_grid(real_B, nrow=5, normalize=True)
        fake_A = make_grid(fake_A, nrow=5, normalize=True)
        fake_B = make_grid(fake_B, nrow=5, normalize=True)
        # Arrange images along y-axis
        image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
        save_image(image_grid, f"sampleimages/{self.dataset_name}/{batches_done}.png", normalize=False)
    
    def add_noise(self, images, epoch):
        """
        Discriminator 입력 이미지에 Gaussian Noise 추가 (500 에폭까지 점진적으로 감소)
        """
        noise_stddev = max(self.initial_noise_stddev * (1 - epoch / self.total_decay_epochs), self.final_noise_stddev)
        noise = torch.randn_like(images) * noise_stddev
        return torch.clamp(images + noise, 0.0, 1.0)  # 픽셀 값이 [0,1] 범위를 유지하도록 클리핑
    
    def train(self, epoch):
        for i, batch in enumerate(self.dataloader):
            # (1) Set model input
            real_A = Variable(batch["A"].type(Tensor)).to(self.device)
            real_B = Variable(batch["B"].type(Tensor)).to(self.device)
            
            # (1.5) Discriminator 입력에 노이즈 추가 (500 에폭까지 점진적으로 감소)
            real_A = self.add_noise(real_A, epoch)
            real_B = self.add_noise(real_B, epoch)

            # (2) Adversarial ground truths - label smoothing
            valid = Variable(Tensor(np.full((real_A.size(0), *self.D_A.output_shape), 0.9)), requires_grad=False).to(self.device)
            fake = Variable(Tensor(np.full((real_A.size(0), *self.D_A.output_shape), 0.1)), requires_grad=False).to(self.device)

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

            # (7) Total loss for Generators
            loss_G = loss_GAN + self.lambda_cyc * loss_cycle + self.lambda_id * loss_identity
            loss_G.backward()
            self.optimizer_G.step()

            # (8) Train Discriminators only if epoch is a multiple of 3
            if epoch % 3 == 0:
                # Discriminator A
                self.optimizer_D_A.zero_grad()
            loss_real_A = self.criterion_GAN(self.D_A(real_A), valid)
            fake_A_ = self.fake_A_buffer.push_and_pop(fake_A)
            loss_fake_A = self.criterion_GAN(self.D_A(fake_A_.detach()), fake)
            loss_D_A = (loss_real_A + loss_fake_A) / 2
            if epoch % 3 == 0:
                loss_D_A.backward()
                self.optimizer_D_A.step()
            
            # Discriminator B
            if epoch % 3 == 0:
                self.optimizer_D_B.zero_grad()
            loss_real_B = self.criterion_GAN(self.D_B(real_B), valid)
            fake_B_ = self.fake_B_buffer.push_and_pop(fake_B)
            loss_fake_B = self.criterion_GAN(self.D_B(fake_B_.detach()), fake)
            loss_D_B = (loss_real_B + loss_fake_B) / 2
            if epoch % 3 == 0:
                loss_D_B.backward()
                self.optimizer_D_B.step()

            loss_D = (loss_D_A + loss_D_B) / 2

            # (9) Print log
            batches_done = epoch * len(self.dataloader) + i
            batches_left = self.n_epochs * len(self.dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - self.prev_time))
            self.prev_time = time.time()

            sys.stdout.write(
                f"\r[Epoch {epoch}/{self.n_epochs}] [Batch {i}/{len(self.dataloader)}] [D loss: {loss_D.item()}] "
                f"[G loss: {loss_G.item()}, adv: {loss_GAN.item()}, cycle: {loss_cycle.item()}, identity: {loss_identity.item()}] "
                f"ETA: {time_left}"
            )

            # 1 Epoch마다 손실 값 저장
            if epoch % 1 == 0:
                self.d_losses.append(loss_D.item())
                self.g_losses.append(loss_G.item())
                self.gan_losses.append(loss_GAN.item())
                self.cycle_losses.append(loss_cycle.item())
                self.identity_losses.append(loss_identity.item())
                self.epochs.append(epoch)

            # (10) If at sample interval save image
            if batches_done % self.sample_interval == 0:
                self.sample_images(batches_done)

        # 1epoch마다 그래프 그리기
        if epoch % 1 == 0:
            self.plot_losses(epoch)

        return loss_G, loss_D


    def plot_losses(self, epoch, save_dir="loss_plots2"):
        # 디렉토리 생성 (없으면)
        os.makedirs(save_dir, exist_ok=True)

        # 손실 값의 평균을 계산
        avg_d_loss = np.mean(self.d_losses) if len(self.d_losses) > 0 else 0
        avg_g_loss = np.mean(self.g_losses) if len(self.g_losses) > 0 else 0
        avg_gan_loss = np.mean(self.gan_losses) if len(self.gan_losses) > 0 else 0
        avg_cycle_loss = np.mean(self.cycle_losses) if len(self.cycle_losses) > 0 else 0
        avg_identity_loss = np.mean(self.identity_losses) if len(self.identity_losses) > 0 else 0

        # epoch에 대한 손실 값 기록
        self.epochs_avg.append(epoch)
        self.d_losses_avg.append(avg_d_loss)
        self.g_losses_avg.append(avg_g_loss)
        self.gan_losses_avg.append(avg_gan_loss)
        self.cycle_losses_avg.append(avg_cycle_loss)
        self.identity_losses_avg.append(avg_identity_loss)

        # 파일 경로 설정
        save_path = os.path.join(save_dir, f"loss_plot_epoch_{epoch}.png")
        
        # 그래프 그리기
        plt.plot(self.epochs_avg, self.d_losses_avg, color='red', label='D Loss')
        plt.plot(self.epochs_avg, self.g_losses_avg, color='blue', label='G Loss')
        plt.plot(self.epochs_avg, self.gan_losses_avg, color='green', label='GAN Loss')
        plt.plot(self.epochs_avg, self.cycle_losses_avg, color='purple', label='Cycle Loss')
        plt.plot(self.epochs_avg, self.identity_losses_avg, color='orange', label='Identity Loss')

        # 그래프 레이블 설정
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Losses During Training')
        plt.legend()

        # 그래프를 PNG 파일로 저장
        plt.savefig(save_path)
        plt.close()
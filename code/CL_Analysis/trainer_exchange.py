from networks import CAEGen, MultiClassDis
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.nn.init as init
import os


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun


class trainer(nn.Module):
    def __init__(self, device,style_dim,optim_para,gen_loss_weight_para,dis_loss_weight_para,train_is):
        super(trainer, self).__init__()


        self.device = device
        self.style_dim = style_dim
        self.gen = CAEGen()
        self.dis_ab = MultiClassDis(device=self.device)
        self.train_is=train_is

        if train_is=='True':
            self.optim_para=optim_para
            self.gen_loss_weight_para=gen_loss_weight_para
            self.dis_loss_weight_para=dis_loss_weight_para

            self.lr = self.optim_para['lr']
            self.lr_step_size = self.optim_para['lr_step_size']
            self.beta1 = self.optim_para['beta1']
            self.beta2 = self.optim_para['beta2']
            self.weight_decay = self.optim_para['weight_decay']

            self.rec_x_w=self.gen_loss_weight_para['rec_x_w']
            self.rec_c_w = self.gen_loss_weight_para['rec_c_w']
            self.rec_s_w = self.gen_loss_weight_para['rec_s_w']
            self.cyc_w = self.gen_loss_weight_para['cyc_w']
            self.gen_dis_w = self.gen_loss_weight_para['gen_dis_w']
            self.gen_cla_w = self.gen_loss_weight_para['gen_cla_w']
            self.dis_dis_w = self.dis_loss_weight_para['dis_dis_w']
            self.dis_cla_w = self.dis_loss_weight_para['dis_cla_w']

            dis_params = list(self.dis_ab.parameters())
            gen_params = list(self.gen.parameters())

            self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                            lr=self.lr, betas=(self.beta1, self.beta2), weight_decay=self.weight_decay)
            self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                            lr=self.lr, betas=(self.beta1, self.beta2), weight_decay=self.weight_decay)

            self.dis_scheduler = lr_scheduler.StepLR(self.dis_opt, step_size=self.lr_step_size,
                                        gamma=0.5, last_epoch=-1)
            self.gen_scheduler = lr_scheduler.StepLR(self.gen_opt, step_size=self.lr_step_size,
                                        gamma=0.5, last_epoch=-1)

            self.apply(weights_init('kaiming'))
            self.dis_ab.apply(weights_init('gaussian'))

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def forward(self, x_a, x_b):
        self.eval()

        c_a, s_a_fake = self.gen.encode(x_a)
        c_b, s_b_fake = self.gen.encode(x_b)
        x_ba = self.gen.decode(c_b, s_a_fake)
        x_ab = self.gen.decode(c_a, s_b_fake)
        self.train()
        return x_ab, x_ba


    def gen_update(self, x_a, x_b,dis_a_real_label,dis_b_real_label):
        self.gen_opt.zero_grad()
        c_a, s_a_prime = self.gen.encode(x_a)
        c_b, s_b_prime = self.gen.encode(x_b)

        x_a_recon = self.gen.decode(c_a, s_a_prime)
        x_b_recon = self.gen.decode(c_b, s_b_prime)

        x_ba = self.gen.decode(c_b, s_a_prime)
        x_ab = self.gen.decode(c_a, s_b_prime)

        c_b_recon, s_a_recon = self.gen.encode(x_ba)
        c_a_recon, s_b_recon = self.gen.encode(x_ab)

        x_aba = self.gen.decode(c_a_recon, s_a_prime)
        x_bab = self.gen.decode(c_b_recon, s_b_prime)

        # reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)

        self.loss_gen_recon_s_a = self.recon_criterion(s_a_recon, s_a_prime)
        self.loss_gen_recon_s_b = self.recon_criterion(s_b_recon, s_b_prime)

        self.loss_gen_recon_c_a = self.recon_criterion(c_a_recon, c_a)
        self.loss_gen_recon_c_b = self.recon_criterion(c_b_recon, c_b)

        self.loss_gen_cycrecon_x_a = self.recon_criterion(x_aba, x_a)
        self.loss_gen_cycrecon_x_b = self.recon_criterion(x_bab, x_b)

        self.dis_a_loss_value_in_gen = self.dis_ab.calc_gen_loss(input_fake=x_ba,real_label=dis_a_real_label,gen_dis_w=self.gen_dis_w,gen_cla_w=self.gen_cla_w)
        self.dis_b_loss_value_in_gen = self.dis_ab.calc_gen_loss(input_fake=x_ab,real_label=dis_b_real_label,gen_dis_w=self.gen_dis_w,gen_cla_w=self.gen_cla_w)

        self.loss_gen_adv_total=1*self.dis_a_loss_value_in_gen+1*self.dis_b_loss_value_in_gen

        self.loss_gen_total = 1 * self.loss_gen_adv_total +self.rec_x_w * self.loss_gen_recon_x_a + self.rec_s_w * self.loss_gen_recon_s_a + self.rec_c_w * self.loss_gen_recon_c_a + \
                              self.rec_x_w * self.loss_gen_recon_x_b + self.rec_s_w * self.loss_gen_recon_s_b + self.rec_c_w * self.loss_gen_recon_c_b + self.cyc_w * self.loss_gen_cycrecon_x_a + self.cyc_w * self.loss_gen_cycrecon_x_b


        self.loss_gen_total.backward()
        self.gen_opt.step()


    def dis_update(self, x_a, x_b,dis_a_real_label,dis_b_real_label):
        self.dis_opt.zero_grad()
        c_a, s_a_prime = self.gen.encode(x_a)
        c_b, s_b_prime = self.gen.encode(x_b)

        x_ba = self.gen.decode(c_b, s_a_prime)
        x_ab = self.gen.decode(c_a, s_b_prime)

        self.loss_dis_a = self.dis_ab.calc_dis_loss(input_fake=x_ba.detach(), input_real=x_a, real_label=dis_a_real_label,dis_dis_w=self.dis_dis_w,dis_cla_w=self.dis_cla_w)
        self.loss_dis_b = self.dis_ab.calc_dis_loss(input_fake=x_ab.detach(), input_real=x_b, real_label=dis_b_real_label,dis_dis_w=self.dis_dis_w,dis_cla_w=self.dis_cla_w)

        self.loss_dis_total = 1 * self.loss_dis_a + 1 * self.loss_dis_b

        self.loss_dis_total.backward()
        self.dis_opt.step()

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()



    def display(self, x_a, x_b):
        self.eval()
        x_a_recon, x_b_recon, x_ba, x_ab = [], [], [], []
        for i in range(x_a.size(0)):

            c_a, s_a_fake = self.gen.encode(x_a[i].unsqueeze(0))
            c_b, s_b_fake = self.gen.encode(x_b[i].unsqueeze(0))
            x_a_recon.append(self.gen.decode(c_a, s_a_fake))
            x_b_recon.append(self.gen.decode(c_b, s_b_fake))

            x_ba.append(self.gen.decode(c_b, s_a_fake.unsqueeze(0)))
            x_ab.append(self.gen.decode(c_a, s_b_fake.unsqueeze(0)))


        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba = torch.cat(x_ba)
        x_ab = torch.cat(x_ab)
        self.train()
        return x_a, x_a_recon, x_b, x_ab, x_b, x_b_recon, x_a, x_ba


    def save(self, save_path, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(save_path, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(save_path, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(save_path, 'optimizer.pt')
        torch.save({'ab': self.gen.state_dict()}, gen_name)
        torch.save({'ab': self.dis_ab.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)


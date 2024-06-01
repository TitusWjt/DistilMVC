from torch.nn.functional import normalize
import torch.nn as nn
import torch

from core.evaulate import evaluation, make_pseudo_label, match, regularization
from core.loss import student_infoNCE, teacher_infoNCE, crossview_contrastive_Loss


class DistilMVC():
    """COMPLETER module."""

    def __init__(self,
                 autoencoder, byol, view):

        self.autoencoder = autoencoder
        self.byol = byol
        self.view = view

    def to_device(self, device):
        """ to cuda if gpu is used """
        self.autoencoder.to(device)
        self.byol.to(device)

    def forward(self, x_list):
        t_list = []
        s_list = []
        z_list = []
        y_list = []
        x_hat_list = []
        for v in range(self.view):
            x = x_list[v]
            z = self.autoencoder.encoders[v](x)
            t = normalize(self.byol.teacher(z), dim=1)
            s = self.byol.student(z)
            y = self.byol.cls(self.byol.student(z))
            x_hat = self.autoencoder.decoders[v](z)
            t_list.append(t)
            z_list.append(z)
            y_list.append(y)
            s_list.append(s)
            x_hat_list.append(x_hat)
        return t_list, y_list, x_hat_list, z_list, s_list

    def predict(self, x_list):
        y_list = []
        pre_list = []
        for v in range(self.view):
            x = x_list[v]
            z = self.autoencoder.encoders[v](x)
            y = self.byol.cls(self.byol.student(z))
            pred = torch.argmax(y, dim=1)
            y_list.append(y)
            pre_list.append(pred)
        return y_list, pre_list

    def train(self, args, data_loader, optimizer, dataset, data_size, class_num):

        #pretrain
        for epoch in range(args.mse_epochs):
            pretrain_mse_loss = 0
            mse = torch.nn.MSELoss()
            for batch_idx, (x_list, _, _) in enumerate(data_loader):
                for v in range(self.view):
                    x_list[v] = x_list[v].to(args.device)
                optimizer.zero_grad()
                t_list, y_list, x_hat_list, z_list, s_list = self.forward(x_list)
                mse_loss_list = []
                for v in range(self.view):
                    mse_loss_list.append(mse(x_list[v], x_hat_list[v]))
                mse_loss = sum(mse_loss_list)
                mse_loss.backward()
                optimizer.step()
                pretrain_mse_loss += mse_loss.item()
            print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(pretrain_mse_loss / len(data_loader)))
        for epoch in range(args.infoNCE_epochs):
            pretrain_loss = 0
            mse = torch.nn.MSELoss()
            for batch_idx, (x_list, _, _) in enumerate(data_loader):
                for v in range(self.view):
                    x_list[v] = x_list[v].to(args.device)
                optimizer.zero_grad()
                t_list, y_list, x_hat_list, z_list, s_list = self.forward(x_list)
                hierarchical_list = []
                for v in range(self.view):
                    for w in range(v + 1, self.view):
                        hierarchical_list.append(teacher_infoNCE(t_list[v], t_list[w], args.batch_size, args.temperature_t))
                        hierarchical_list.append(student_infoNCE(y_list[v], y_list[w], class_num, args.temperature_s))
                        hierarchical_list.append(
                            args.iic_factor * crossview_contrastive_Loss(torch.softmax(z_list[v], dim=0),
                                                                torch.softmax(z_list[w], dim=0)))
                    hierarchical_list.append(mse(x_list[v], x_hat_list[v]))
                hierarchical_loss = sum(hierarchical_list)
                hierarchical_loss.backward()
                optimizer.step()
                self.byol.update_moving_average(args.mse_epochs+epoch - 199 + batch_idx, args.mse_epochs+ args.infoNCE_epochs- 199)
                pretrain_loss += hierarchical_loss.item()
            print('Epoch {}'.format(args.mse_epochs+epoch), 'Loss:{:.6f}'.format(pretrain_loss / len(data_loader)))
        acc, nmi, pur = evaluation(self, dataset, self.view, args.device, data_size)

        # fine-tuning
        new_pseudo_label = make_pseudo_label(self, args.device, dataset, self.view, class_num, data_size)
        for epoch in range(args.self_distil):
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=data_size,
                shuffle=False,
            )
            tot_loss = 0.
            KL = torch.nn.KLDivLoss()
            for batch_idx, (x_list, _, idx) in enumerate(loader):
                for v in range(self.view):
                    x_list[v] = x_list[v].to(args.device)
                optimizer.zero_grad()
                _, y_list, _, _, s_list = self.forward(x_list)
                loss_list = []
                for v in range(self.view):
                    p = new_pseudo_label[v].numpy().T[0]
                    with torch.no_grad():
                        y = y_list[v].detach().cpu()
                        y = torch.argmax(y, dim=1).numpy()
                        p_hat = match(p, y, args.device)
                        dark_konwledge = torch.Tensor(regularization(p_hat, class_num, args.temperature_d)).to(args.device)
                    loss_list.append(KL(torch.log(y_list[v]), dark_konwledge))
                    # loss_list.append(cross_entropy(qs[v], p_hat))
                loss = sum(loss_list)
                loss.backward()
                optimizer.step()
                tot_loss += loss.item()
            print('Epoch {}'.format(args.mse_epochs+epoch+args.infoNCE_epochs), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))
        acc, nmi, pur = evaluation(self, dataset, self.view, args.device, data_size)




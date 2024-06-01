
from core.evaulate import evaluation
from utils import create_dict, recode


def pretrain(model, Pretrain_p, dataset, data_loader, criterion, view, optimizer, data_size, class_num, device, tb_writer):
    #record
    res_excel = create_dict(view)

    #train
    for epoch in range(Pretrain_p['p_epoch']):
        loss_tot, loss_rec, loss_iic = 0, 0, 0
        for batch_idx, (xs, y, idx) in enumerate(data_loader):
            for v in range(view):
                xs[v] = xs[v].to(device)
            zs, rs = model(xs)
            tot_list , rec_list, iic_list= [], [], []
            for v in range(view):
                rec_list.append(criterion.mse(xs[v], rs[v]))
                for w in range(v+1, view):
                    iic_list.append(criterion.forward_iic(zs[v], zs[w]))
            rec = sum(rec_list)
            iic = sum(iic_list)
            tot = rec + iic
            optimizer.zero_grad()
            tot.backward()
            optimizer.step()
            loss_rec += rec.item()
            loss_iic += iic.item()
            loss_tot += tot.item()
        output = "Epoch : {:.0f}/{:.0f} ===> Reconstruction loss = {:.4f}" \
                 "===> IIC loss = {:.4e} ===> Total Loss = {:.4e}" \
            .format((epoch + 1), Pretrain_p['p_epoch'], loss_rec, loss_iic, loss_tot)
        print(output)
        if (epoch+1) % Pretrain_p['p_interval'] == 0:
            scores_each, scores_tot = evaluation(Pretrain_p, model, dataset, view, data_size, class_num, device)
            tags = ["train_loss", "accuracy", "learning_rate"]
            tb_writer.add_scalar(tags[0], loss_tot, epoch)
            tb_writer.add_scalar(tags[1], scores_tot['kmeans']['accuracy'], epoch)
            tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)
            recode(res_excel, epoch + 1, Pretrain_p['p_epoch'], loss_tot, scores_tot, scores_each)
        # save result
        # if epoch == x -1:
        #  torch.save(model.state_dict(), Pretrain_p['pretrain_path'])

    #pd.DataFrame(res_excel).to_excel(Pretrain_p['pretrain_save'])






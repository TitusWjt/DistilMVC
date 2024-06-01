from sklearn.metrics import v_measure_score, adjusted_rand_score, accuracy_score
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
import numpy as np
import torch

from core.evaulate_method import cluster_acc, purity, get_cluster_sols, get_y_preds, clustering_metric

def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    u = linear_sum_assignment(w.max() - w)
    ind = np.concatenate([u[0].reshape(u[0].shape[0], 1), u[1].reshape([u[0].shape[0], 1])], axis=1)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def purity(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return accuracy_score(y_true, y_voted_labels)

def vaild(label, pred):
    nmi = v_measure_score(label, pred)
    ari = adjusted_rand_score(label, pred)
    acc = cluster_acc(label, pred)
    pur = purity(label, pred)
    return dict({'acc': acc, 'nmi': nmi, 'ari': ari, 'pur': pur})

def evaluate(label, pred):
    nmi = v_measure_score(label, pred)
    ari = adjusted_rand_score(label, pred)
    acc = cluster_acc(label, pred)
    pur = purity(label, pred)
    return nmi, ari, acc, pur


def evaluation(self, dataset, view, device, data_size,):
    test_loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
    )
    self.autoencoder.eval()
    self.byol.eval()
    soft_vector = []
    pred_vectors = []
    T_list = []
    Z_list = []
    for v in range(view):
        pred_vectors.append([])
        T_list.append([])
        Z_list.append([])
    labels_vector = []

    for step, (x_list, y, idx) in enumerate(test_loader):
        for v in range(view):
            x_list[v] = x_list[v].to(device)
        with torch.no_grad():
            y_list, pre_list = self.predict(x_list)
            t_list, _, _, z_list, s_list = self.forward(x_list)
            ys = sum(y_list) / view
        for v in range(view):
            t_list[v] = t_list[v].detach()
            z_list[v] = z_list[v].detach()
            pre_list[v] = pre_list[v].detach()
            pred_vectors[v].extend(pre_list[v].cpu().detach().numpy())
            T_list[v].extend(t_list[v].cpu().detach().numpy())
            Z_list[v].extend(z_list[v].cpu().detach().numpy())
        ys = ys.detach()
        soft_vector.extend(ys.cpu().detach().numpy())
        labels_vector.extend(y.numpy())
    labels_vector = np.array(labels_vector).reshape(data_size)
    total_pred = np.argmax(np.array(soft_vector), axis=1)
    for v in range(view):
        T_list[v] = np.array(T_list[v])
        Z_list[v] = np.array(Z_list[v])
        pred_vectors[v] = np.array(pred_vectors[v])
    high_level_vectors,  low_level_vectors = T_list,  Z_list

    # print("Clustering results on common features of each view:")
    # scores_each = []
    # for v in range(view):
    #     scores_each.append([])
    #     scores = evaluate([Zs[v]], labels_vector)
    #     print('ACC{} = {:.4f} NMI{} = {:.4f} ARI{} = {:.4f} PUR{}={:.4f}'.format(v + 1, scores['kmeans']['accuracy'],
    #                                                                              v + 1, scores['kmeans']['NMI'],
    #                                                                              v + 1, scores['kmeans']['ARI'],
    #                                                                              v + 1, scores['kmeans']['pur']))
    #     scores_each[v].append(scores['kmeans']['accuracy'])
    #     scores_each[v].append(scores['kmeans']['NMI'])
    #     scores_each[v].append(scores['kmeans']['ARI'])
    #     scores_each[v].append(scores['kmeans']['pur'])
    #
    #
    # print("Clustering results on common features of all view:")
    # latent_fusion = np.concatenate(Zs,axis=1)
    # scores_tot = evaluate([latent_fusion], labels_vector)
    # print('ACC = {:.4f} NMI = {:.4f} ARI = {:.4f} PUR={:.4f}'.format( scores_tot['kmeans']['accuracy'],
    #                                                                   scores_tot['kmeans']['NMI'],
    #                                                                   scores_tot['kmeans']['ARI'],
    #                                                                   scores_tot['kmeans']['pur']))
    print("Clustering results on semantic labels: " + str(labels_vector.shape[0]))
    nmi, ari, acc, pur = evaluate(labels_vector, total_pred)
    print('ACC = {:.4f} NMI = {:.4f} ARI = {:.4f} PUR={:.4f}'.format(acc, nmi, ari, pur))

    return acc, nmi, pur

from sklearn.preprocessing import MinMaxScaler
def make_pseudo_label(model, device, dataset, view, class_num, data_size):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=data_size,
        shuffle=False,
    )
    model.autoencoder.eval()
    model.byol.eval()
    scaler = MinMaxScaler()
    for step, (x_list, y, _) in enumerate(loader):
        for v in range(view):
            x_list[v] = x_list[v].to(device)
        with torch.no_grad():
            t_list, _, _, _,s_list = model.forward(x_list)
        for v in range(view):
            t_list[v] = t_list[v].cpu().detach().numpy()
            t_list[v] = scaler.fit_transform(t_list[v])

    kmeans = KMeans(n_clusters=class_num, n_init=100)
    new_pseudo_label = []
    for v in range(view):
        Pseudo_label = kmeans.fit_predict(t_list[v])
        Pseudo_label = Pseudo_label.reshape(data_size, 1)
        Pseudo_label = torch.from_numpy(Pseudo_label)
        new_pseudo_label.append(Pseudo_label)

    return new_pseudo_label


def match(y_true, y_pred, device):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    new_y = np.zeros(y_true.shape[0])
    for i in range(y_pred.size):
        for j in row_ind:
            if y_true[i] == col_ind[j]:
                new_y[i] = row_ind[j]
    new_y = torch.from_numpy(new_y).long().to(device)
    new_y = new_y.view(new_y.size()[0])
    return new_y

def regularization(label, class_num ,epsilon):
    import numpy as np
    results = np.zeros((len(label), class_num))
    for i, label in enumerate(label):
        results[i, label] = 1.
    soft_label = ((1-epsilon)*results) + epsilon/class_num
    return soft_label
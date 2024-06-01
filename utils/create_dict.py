def create_dict(v):
    dict = {}
    dict['epoch'] = []
    dict['loss'] = []
    dict['acc'] = []
    dict['nmi'] = []
    dict['ari'] = []
    dict['pur'] = []
    for i in range(v):
        i = str(i)
        dict['acc' + i] = []
        dict['nmi' + i] = []
        dict['ari' + i] = []
        dict['pur' + i] = []
    return dict

def recode(dict, epoch, epoch_max, loss_tot, scores_tot, scores_each):
    dict['epoch'].append(epoch)
    dict['loss'].append(loss_tot)
    dict['acc'].append(scores_tot['kmeans']['accuracy'])
    dict['nmi'].append(scores_tot['kmeans']['NMI'])
    dict['ari'].append(scores_tot['kmeans']['ARI'])
    dict['pur'].append(scores_tot['kmeans']['pur'])
    for i in range(len(scores_each)):
        j = str(i)
        dict['acc' + j].append(scores_each[i][0])
        dict['nmi' + j].append(scores_each[i][1])
        dict['ari' + j].append(scores_each[i][2])
        dict['pur' + j].append(scores_each[i][3])
    if epoch == epoch_max:
        print('mean acc is' + ' ' + str(sum(dict['acc']) / len(dict['acc'])))
        print('max acc is' + ' ' + str(max(dict['acc'])))
        print('epoch of max acc is' + ' ' + str(dict['acc'].index(max(dict['acc']))+1))








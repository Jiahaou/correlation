from pathlib import Path
import torch
from tqdm import tqdm
from pif.hvp_grad import grad_z
from pif.utils import save_json
from torch.autograd import grad
from collections import OrderedDict
from torch.nn import functional as F, CrossEntropyLoss
import time
from src.compress import compress_pkl
import zipfile
import pickle
import _pickle as cPickle
import tensorflow as tf
import gzip
import datatable
def calc_all_grad(config, model, train_loader, test_loader,
                  ntest_start, ntest_end, mode='TC'):
    """Calculates the influence function by first calculating
    all grad_z, all s_test and then loading them to calc the influence"""

    '''
    config['outdir'] specifies the model name, i.e. different model saves at different out dir with unique name/id/

    '''
    depth, r = config['recursion_depth'], config['r_averaging']

    outdir = Path(config["outdir"])

    # breakpoint()
    if not outdir.exists():
        outdir.mkdir(parents=True, exist_ok=True)

    influence_results = {}

    ntrainiter = len(train_loader.dataset)
    ntest_end = len(test_loader.dataset)
    model.eval()
    grad_z_test = ()
    start=time.perf_counter()
    save_train=False#/home/njtech/Python-projects/sinan-local-master
    save_test=False
    change=True#do you want to change data from 2 diam to 1 diam?
    need=False#do you need calculate directly????? if you have data ,you needn't and change it "False".
    #testr = open("../compress/test.pkl","wb")
    if change==True:
        compress_pkl("../compress/train_10.pkl","xiangliang_demen.pkl","train_10-1.pkl",39)
        compress_pkl("../compress/test_10.pkl", "xiangliangtest_demen.pkl", "test_10-1.pkl",39)
    if save_train==True:
        text = open("../compress/train_demen.pkl", "wb")
        for j, batch_t in enumerate(tqdm(train_loader)):  # in tqdm(range(ntrainiter)):
            if torch.cuda.is_available():
                device = torch.device(f"cuda:0")
            else:
                device = torch.device("cpu")
            input_ids_t, token_type_ids_t, lm_labels_t = batch_t
            input_ids_t, token_type_ids_t, lm_labels_t = \
                input_ids_t.to(device), token_type_ids_t.to(device), lm_labels_t.to(device)
            ti, tt, tl, t_idx, td = train_loader.dataset[j]
            t_idx = int(t_idx)
            outputs = model(
                input_ids=input_ids_t,
                token_type_ids=token_type_ids_t,
                labels=lm_labels_t
            )
            loss_t, logits = outputs[0], outputs[1]
            # grad_z_train = grad_z(input_ids_t, token_type_ids_t, lm_labels_t, model, device, pid)
            grad_z_train = grad(loss_t, model.parameters())  # , allow_unused=True)
            grad_z_train = pick_gradient(grad_z_train, model)  # pick effective ones           #list have 148 tensor
            cPickle.dump(grad_z_train, text)
            continue
    if save_test==True:
        testr = open("../compress/test_demen.pkl", "wb")
        for i, batch in enumerate(tqdm(test_loader)):
            if torch.cuda.is_available():
                device = torch.device(f"cuda:0")
            else:
                device = torch.device("cpu")
            input_ids, token_type_ids, lm_labels = batch
            input_ids, token_type_ids, lm_labels = \
                input_ids.to(device), token_type_ids.to(device), lm_labels.to(device)
            di, dt, dl, idx, dd = test_loader.dataset[i]
            idx = int(idx)

            outputs = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                labels=lm_labels
            )
            loss, logits = outputs[0], outputs[1]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous().to(device)
            pad_id = -100
            loss_fct = CrossEntropyLoss(ignore_index=pad_id, reduction='sum')
            siz = shift_logits.size(-1)
            log = shift_logits.view(-1, siz)
            lab = shift_labels.view(-1)
            loss = loss_fct(log, lab)


            grad_z_test = grad(loss, model.parameters())  # , allow_unused=True)


            grad_z_test = pick_gradient(grad_z_test, model)  # pick effective ones
            cPickle.dump(grad_z_test, testr)
            # continue or not can take test data or not
            continue
    # for i in tqdm(range(ntest_start, ntest_end)):
    testr = open("../compress/test_demen.pkl", "rb")
    testr1 = open("test_demen.pkl", "rb")
    for i, batch in enumerate(tqdm(test_loader)):
        if torch.cuda.is_available():
            device = torch.device(f"cuda:0")
        else:
            device = torch.device("cpu")

        input_ids, token_type_ids, lm_labels = batch
        input_ids, token_type_ids, lm_labels = \
            input_ids.to(device), token_type_ids.to(device), lm_labels.to(device)
        di, dt, dl, idx, dd = test_loader.dataset[i]
        idx = int(idx)

        if outdir.joinpath(f'did-{idx}.{mode}.json').exists():
            continue
        if mode =='TC':
            if need==True:
                outputs = model(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    labels=lm_labels
                )
                loss, logits = outputs[0], outputs[1]

                device = torch.device(f"cuda:0")
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = lm_labels[..., 1:].contiguous().to(device)
                pad_id = -100
                loss_fct = CrossEntropyLoss(ignore_index=pad_id, reduction='sum')
                siz = shift_logits.size(-1)
                log = shift_logits.view(-1, siz)
                lab = shift_labels.view(-1)
                loss = loss_fct(log, lab)

                start = time.clock()
                grad_z_test = grad(loss, model.parameters())  # , allow_unused=True)
                end1 = time.clock()

                grad_z_test = pick_gradient(grad_z_test, model)  # pick effective ones
                cPickle.dump(grad_z_test,testr)
                #continue or not can take test data or not
                #continue
                end2 = time.clock()
                # print(end2-end1)
            else:
                grad_z_test = cPickle.load(testr1)
                #grad_z_test = cPickle.load(testr1)*cPickle.load(3_xiangliang)

        if mode == 'IF':
            s_test = torch.load(config['stest_path'] + f"/did-{int(idx)}_recdep{depth}_r{r}.s_test")
            s_test = [s_t.cuda() for s_t in s_test]

        train_influences = {}
        # if i==0:
        #     text = open("../compress/yasuo.pkl", 'wb')
        #     cal_grad_z_train(model,text,train_loader,mode)
        #     text.close()

        if need==True:
            #text=open("2_inverse.pkl","wb")#100tiao
            text=open("../compress/yasuo_10.pkl","rb")

            for j, batch_t in enumerate(tqdm(train_loader)):  # in tqdm(range(ntrainiter)):

                input_ids_t, token_type_ids_t, lm_labels_t = batch_t
                input_ids_t, token_type_ids_t, lm_labels_t = \
                    input_ids_t.to(device), token_type_ids_t.to(device), lm_labels_t.to(device)
                ti, tt, tl, t_idx, td = train_loader.dataset[j]
                t_idx = int(t_idx)
                outputs = model(
                    input_ids=input_ids_t,
                    token_type_ids=token_type_ids_t,
                    labels=lm_labels_t
                )
                loss_t, logits = outputs[0], outputs[1]
                # grad_z_train = grad_z(input_ids_t, token_type_ids_t, lm_labels_t, model, device, pid)
                grad_z_train = grad(loss_t, model.parameters())  # , allow_unused=True)
                grad_z_train = pick_gradient(grad_z_train, model)  # pick effective ones           #list have 148 tensor
                #cPickle.dump(grad_z_train, text)

                score = 0
                if mode == 'IF':
                    score = param_vec_dot_product(s_test, grad_z_train)
                elif mode == 'TC':
                    score = param_vec_dot_product(grad_z_test, grad_z_train)

                # breakpoint()

                if t_idx not in train_influences:
                    train_influences[t_idx] = {'train_dat': (td),
                                              'if': float(score)}

            text.close()
        else:

            #text1=open("../compress/yasuo.pkl","rb")
            text1 = open("train_demen.pkl", 'rb')#100
            # i=0
            for j, batch_t in enumerate(tqdm(train_loader)):  # in tqdm(range(ntrainiter)):
                # if j == 0:
                #     continue
                input_ids_t, token_type_ids_t, lm_labels_t = batch_t
                input_ids_t, token_type_ids_t, lm_labels_t = \
                    input_ids_t.to(device), token_type_ids_t.to(device), lm_labels_t.to(device)
                ti, tt, tl, t_idx, td = train_loader.dataset[j]
                t_idx = int(t_idx)
                # outputs = model(
                #     input_ids=input_ids_t,
                #     token_type_ids=token_type_ids_t,
                #     labels=lm_labels_t
                # )
                # loss_t, logits = outputs[0], outputs[1]
                # #grad_z_train = grad_z(input_ids_t, token_type_ids_t, lm_labels_t, model, device, pid)
                # grad_z_train = grad(loss_t, model.parameters())  # , allow_unused=True)
                # grad_z_train = pick_gradient(grad_z_train, model)  # pick effective ones

                grad_z_train = cPickle.load(text1)#this is a list
                #grad_z_train = cPickle.load(text1)*cPickle.load(3_xiangliang1)

                # i+=1

                score = 0
                if mode == 'IF':
                    score = param_vec_dot_product(s_test, grad_z_train)
                elif mode == 'TC':
                    score = param_vec_dot_product(grad_z_test, grad_z_train)

                # breakpoint()

                if t_idx not in train_influences:
                    train_influences[t_idx] = {'train_dat': (td),
                                               'if': float(score)}
                # print(i)


            # with open("1_compross.pkl", 'rb')as text1:
            #     with open("3_xiangliang.pkl","rb")as text2:
            #         s1 = time.clock()
            #
            #         for j, batch_t in enumerate(train_loader):  # in tqdm(range(ntrainiter)):
            #             # if j == 0:
            #             #     continue
            #             input_ids_t, token_type_ids_t, lm_labels_t = batch_t
            #             input_ids_t, token_type_ids_t, lm_labels_t = \
            #                 input_ids_t.to(device), token_type_ids_t.to(device), lm_labels_t.to(device)
            #             ti, tt, tl, t_idx, td = train_loader.dataset[j]
            #             t_idx = int(t_idx)
            #             # outputs = model(
            #             #     input_ids=input_ids_t,
            #             #     token_type_ids=token_type_ids_t,
            #             #     labels=lm_labels_t
            #             # )
            #             # loss_t, logits = outputs[0], outputs[1]
            #             # grad_z_train = grad_z(input_ids_t, token_type_ids_t, lm_labels_t, model, device, pid)
            #             # grad_z_train = grad(loss_t, model.parameters())  # , allow_unused=True)
            #             # grad_z_train = pick_gradient(grad_z_train, model)  # pick effective ones
            #             print(cPickle.load(text1)[0])
            #             a1=cPickle.load(text1)
            #             a2=cPickle.load(text2).type_as(a1)
            #             grad_z_train = []
            #             [grad_z_train.append(a1[i]) if a1[i].ndim == 1 else grad_z_train.append(torch.matmul(torch.matmul(a2[i], a1[i].T), a1[i])) for i in range(len(a1))]
            #             score = 0
            #             if mode == 'IF':
            #                 score = param_vec_dot_product(s_test, grad_z_train)
            #             elif mode == 'TC':
            #                 score = param_vec_dot_product(grad_z_test, grad_z_train)
            #
            #             # breakpoint()
            #
            #             if t_idx not in train_influences:
            #                 train_influences[t_idx] = {'train_dat': (td),
            #                                            'if': float(score)}
            #         text1.close()
            #         s2 = time.clock()
            #         print(s2 - s1)

        # train_influences1 = {}
        # train_influences1 = OrderedDict(sorted(train_influences, key=lambda x: x[1]['if'], reverse=True))#, reverse=True
        if idx not in influence_results:
            influence_results[idx] = {'test_dat': (dd),
                                      'ifs': train_influences}
        save_json(influence_results, outdir.joinpath(f'did-{idx}.{mode}.json'))

    end1=time.perf_counter()
    print(end1-start)
#def drop_tokens(embeddings,word_dropout):


def param_vec_dot_product(a, b):
    """ dot product between two lists"""
    return sum([torch.dot(at.flatten(), bt.flatten()) for at, bt in zip(a, b)])
    # breakpoint()
def cal_grad_z_train(model,text,train_loader,mode="TC"):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:0")
    else:
        device = torch.device("cpu")
    for j, batch_t in enumerate(train_loader):  # in tqdm(range(ntrainiter)):

        input_ids_t, token_type_ids_t, lm_labels_t = batch_t
        input_ids_t, token_type_ids_t, lm_labels_t = \
            input_ids_t.to(device), token_type_ids_t.to(device), lm_labels_t.to(device)
        ti, tt, tl, t_idx, td = train_loader.dataset[j]
        t_idx = int(t_idx)
        outputs = model(
            input_ids=input_ids_t,
            token_type_ids=token_type_ids_t,
            labels=lm_labels_t
        )
        loss_t, logits = outputs[0], outputs[1]
        # grad_z_train = grad_z(input_ids_t, token_type_ids_t, lm_labels_t, model, device, pid)
        grad_z_train = grad(loss_t, model.parameters())  # , allow_unused=True)
        grad_z_train = pick_gradient(grad_z_train, model)  # pick effective ones
        pickle.dump(grad_z_train, text)



def pick_gradient(grads, model):
    """
    pick the gradients by name.
    Specifically for BERTs, it extracts 10, 11 layer, pooler and classification layers params.
    """
    return [grad for grad, (n, p) in zip(grads, model.named_parameters())]
            # if 'layer.10.' in n or 'layer.11.' in n
            # or 'classifier.' in n or 'pooler.' in n




import os
import torch.optim as optim
import torch.utils.data
from UDA.data_UDA import DATA
from UDA.model_UDA import Model
import numpy as np
from UDA.test_UDA import test
from UDA.parser_2 import arg_parse
import UDA.model_UDA as models

def conv2d(m,n,k,act=True):
    layers =  [torch.nn.Conv2d(m,n,k,padding=1)]

    if act: layers += [torch.nn.ELU()]

    return torch.nn.Sequential(
        *layers
    )

if __name__ == '__main__':
    #load args
    args = arg_parse()
    #function to save model
    def save_model(model, save_path):
        torch.save(model.state_dict(), save_path)

    def two_train(source, target, source_name, target_name, save_name):
        print("Starting training DANN", source_name, "->", target_name)
        len_dataloader = min(len(source), len(target))
        num_print = int(len_dataloader/10)
        best = 0
        err_sl_mean = 0
        err_sd_mean = 0
        err_mean = 0
        for epoch in range(args.epoch):
            for i, ((images_s, id_s), (images_t, _)) in enumerate(zip(source, target)):
                images_s = images_s.cuda()
                id_s = id_s.view(-1).cuda()
                images_t = images_t.cuda()

                #train the model
                my_model.zero_grad()

                domain_s, class_s = my_model(images_s)
                domain_t, _ = my_model(images_t)

                err_s_class = loss_class(class_s, id_s).mean()
                err_domain = loss_domain(domain_s, domain_t, id_s).mean()
                err = err_s_class + err_domain

                err.backward()
                optimizer.step()
                err_sl_mean += err_s_class.cpu().item()
                err_sd_mean += err_domain.cpu().item()
                err_mean += err.cpu().item()
                if (i+1)%num_print == 0:
                    print('epoch: %d/%d, [iter: %d / all %d], err_s_label: %f,  err_domain: %f, err_total: %f,' \
                    % (epoch+1, args.epoch, i+1, len_dataloader, err_sl_mean/num_print,err_sd_mean/num_print,err_mean/num_print))
                    err_sl_mean = 0
                    err_sd_mean = 0
                    err_mean = 0
            #_ = test(source_name, epoch, args, my_model)
            accu = test(target_name, epoch+1, args, my_model)
            if accu > best:
                save_model(my_model, os.path.join(args.save_dir, save_name+'.pth.tar'))
                best = accu
        print("Finished training DANN", source_name, "->", target_name)
        return best

    #Params
    mni_name = 'mnistm'
    svhn_name = 'svhn'

    #load training datasets
    mni = torch.utils.data.DataLoader(DATA(args=args, type = mni_name, mode = 'train'),
                                                 batch_size=args.train_batch,
                                                 num_workers=args.workers,
                                                 shuffle=True)

    svhn = torch.utils.data.DataLoader(DATA(args=args, type=svhn_name, mode='train'),
                                         batch_size=args.train_batch,
                                         num_workers=args.workers,
                                         shuffle=True)

    #Create model, optimizer and losses
    my_model = Model().cuda()
    optimizer = optim.Adam(my_model.parameters(), lr=args.lr, betas=(0.5, 0.999), amsgrad=True)
    loss_class = torch.nn.CrossEntropyLoss().cuda()
    loss_domain = models.AssociativeLoss().cuda()
    for p in my_model.parameters():
        p.requires_grad = True

    print("starting trainign...")

    #train DANN mnistm->svhn
    #best_error = two_train(mni, svhn, mni_name, svhn_name, 'model_UDA_mnistm-svhn')

    # train DANN svhn->mnistm
    best_error = two_train(svhn, mni, svhn_name, mni_name, 'model_UDA_svhn-mnistm')

    print("Best obtained Accuracy: ", best_error)



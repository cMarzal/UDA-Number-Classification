import os
import torch.optim as optim
import torch.utils.data
from DANN.data_DANN import DATA
from DANN.model_DANN import CNNModel
import numpy as np
from DANN.test_DANN import test
from DANN.parser_2 import arg_parse


if __name__ == '__main__':

    # load args
    args = arg_parse()

    # function to save model
    def save_model(model, save_path):
        torch.save(model.state_dict(), save_path)

    # function to train model in one source and run test on given target
    def one_train(source, source_name, target_name, save_name):
        print("Starting training on", source_name, "to maximize accuracy on", target_name)
        len_dataloader = len(source)
        num_print = int(len_dataloader/10)
        best = 0
        err_mean = 0
        for epoch in range(args.epoch):
            for i, (images_s, id_s) in enumerate(source):
                # set alpha parameter
                alpha = 0
                # images to cuda
                images_s = images_s.cuda()
                id_s = id_s.view(-1).cuda()
                batch_size = images_s.size(0)
                #run model
                my_model.zero_grad()
                class_s, _, _ = my_model(images_s, alpha)
                err_s_label = loss_class(class_s, id_s)
                err_s_label.backward()
                optimizer.step()
                err_mean += err_s_label.cpu().item()
                if (i+1)%num_print == 0:
                    print('epoch: %d/%d, [iter: %d / all %d], err_s_label: %f' \
                    % (epoch, args.epoch, i+1, len_dataloader, err_mean/num_print))
                    err_mean = 0
            accu = test(target_name, epoch+1, args, my_model)
            if accu > best:
                best = accu

        print("Finished training on", source_name, "to maximize accuracy on", target_name)
        return best

    # Function to run model from one source and the domain of the target and then run test on target
    def two_train(source, target, source_name, target_name, save_name):
        print("Starting training DANN", source_name, "->", target_name)
        len_dataloader = min(len(source), len(target))
        num_print = int(len_dataloader/10)
        best = 0
        err_sl_mean = 0
        err_sd_mean = 0
        err_t_mean = 0
        err_mean = 0
        for epoch in range(args.epoch):
            for i, ((images_s, id_s), (images_t, _)) in enumerate(zip(source, target)):
                p = float(i + epoch * len_dataloader) / args.epoch / len_dataloader
                alpha = 2. / (1. + np.exp(-10 * p)) - 1
                images_s = images_s.cuda()
                id_s = id_s.view(-1).cuda()
                images_t = images_t.cuda()
                batch_size = images_s.size(0)
                domain_label = torch.zeros(batch_size)
                domain_label = domain_label.long().cuda()

                # train the model
                my_model.zero_grad()
                class_s, domain_s, _ = my_model(images_s, alpha)
                err_s_class = loss_class(class_s, id_s).mean()
                err_s_domain = loss_domain(domain_s, domain_label).mean()

                domain_label = torch.ones(batch_size)
                domain_label = domain_label.long().cuda()
                _, domain_t, _ = my_model(images_t, alpha)
                err_t_domain = loss_domain(domain_t, domain_label).mean()
                err = err_s_class + err_s_domain + err_t_domain
                err.backward()
                optimizer.step()
                err_sl_mean += err_s_class.cpu().item()
                err_sd_mean += err_s_domain.cpu().item()
                err_t_mean += err_t_domain.cpu().item()
                err_mean += err.cpu().item()
                if (i+1)%num_print == 0:
                    print('epoch: %d/%d, [iter: %d / all %d], err_s_label: %f,  err_s_domain: %f, err_t_domain: %f, err_total: %f,' \
                    % (epoch+1, args.epoch, i+1, len_dataloader, err_sl_mean/num_print,err_sd_mean/num_print,err_t_mean/num_print,err_mean/num_print))
                    err_sl_mean = 0
                    err_sd_mean = 0
                    err_t_mean = 0
                    err_mean = 0
            # = test(source_name, epoch, args, my_model)
            accu = test(target_name, epoch+1, args, my_model)
            if accu > best:
                save_model(my_model, os.path.join(args.save_dir, save_name+'.pth.tar'))
                best = accu
        save_model(my_model, os.path.join(args.save_dir, save_name + '_last.pth.tar'))
        print("Finished training DANN", source_name, "->", target_name)
        return best

    #Params
    mni_name = 'mnistm'
    svhn_name = 'svhn'

    # load training datasets
    mni = torch.utils.data.DataLoader(DATA(args=args, type = mni_name, mode = 'train'),
                                                 batch_size=args.train_batch,
                                                 num_workers=args.workers,
                                                 shuffle=True)

    svhn = torch.utils.data.DataLoader(DATA(args=args, type=svhn_name, mode='train'),
                                         batch_size=args.train_batch,
                                         num_workers=args.workers,
                                         shuffle=True)

    # Create model, optimizer and losses
    my_model = CNNModel().cuda()
    optimizer = optim.Adam(my_model.parameters(), lr=args.lr)
    loss_class = torch.nn.NLLLoss().cuda()
    loss_domain = torch.nn.NLLLoss().cuda()

    for p in my_model.parameters():
        p.requires_grad = True

    print("starting trainign...")

    # train mnistm only-max svhn
    #best_error = one_train(mni, mni_name, svhn_name, "model_mnistm-svhn")

    # train mnistm only-max mnistm
    #best_error = one_train(mni, mni_name, mni_name, "model_mnistm-mnistm")

    # train svhn only-max mnistm
    #best_error = one_train(svhn, svhn_name, mni_name, "model_svhn-mnistm")

    # train svhn only-max svhn
    best_error = one_train(svhn, svhn_name, svhn_name, "model_svhn-svhn")

    # train DANN mnistm->svhn
    #best_error = two_train(mni, svhn, mni_name, svhn_name, 'model_DANN_mnistm-svhn')

    # train DANN svhn->mnistm
    #best_error = two_train(svhn, mni, svhn_name, mni_name, 'model_DANN_svhn-mnistm')

    print("Best obtained Accuracy: ", best_error)



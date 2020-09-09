import torch
import numpy as np
import pandas as pd
from DANN.data_DANN2 import DATA
from UDA.model_UDA import Model
from UDA.parser_2 import arg_parse


if __name__ == '__main__':
    args = arg_parse()

    def arrange_data(dataset, model):
        classes = 0
        labels = 0
        for i, (images_s, id_s) in enumerate(dataset):
            images_s = images_s.cuda()
            id_s = id_s
            batch_size = images_s.size(0)
            _ , cl= model(images_s)
            cl = cl.data.max(1, keepdim=True)[1]
            if i == 0:
                classes = cl.cpu().detach().numpy()
                labels = np.array(id_s)

            else:
                classes = np.concatenate((classes, cl.cpu().detach().numpy()))
                labels = np.concatenate((labels, np.array(id_s)))

        return classes, labels

    def getdata(args):
        dataset = torch.utils.data.DataLoader(DATA(args.dir_img),
                                              batch_size=100,
                                              num_workers=args.workers,
                                              shuffle=False)
        return dataset

    dataset = getdata(args)

    if args.target == 'mnistm':
        model_state = 'model_UDA_svhn-mnistm.pth.tar'
    else:
        model_state = 'model_UDA_mnistm-svhn.pth.tar'

    my_model = Model().eval().cuda()
    my_model.load_state_dict(torch.load(model_state))
    classes, dirs = arrange_data(dataset, my_model)

    df = pd.DataFrame()
    df['image_name'] = dirs
    df['label'] = classes

    df.to_csv(args.save_csv, index = False)
import torch.utils.data
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from DANN.data_DANN import DATA
from DANN.model_DANN import CNNModel
from DANN.parser_2 import arg_parse
from sklearn.manifold import TSNE
import pandas as pd

if __name__ == '__main__':

    args = arg_parse()


    def arrange_data(dataset, model):
        features = np.empty([1, 50, 4, 4])
        labels = np.empty([1])
        domains = np.empty([1])
        for i, ((images_s, id_s), (images_t, id_t)) in enumerate(dataset):
            if i < 7:
                alpha = 0
                images_s = images_s.cuda()
                id_s = id_s.view(-1)
                images_t = images_t.cuda()
                id_t = id_t.view(-1)
                batch_size = images_s.size(0)

                _, _, features1 = model(images_s, alpha)
                _, _, features2 = model(images_t, alpha)

                domain1 = np.ones((batch_size), dtype=int)
                domain2 = np.zeros((batch_size), dtype=int)

                if i == 0:
                    domains = domain1
                    domains = np.concatenate((domains, domain2))

                    labels = np.array(id_s)
                    labels = np.concatenate((labels, np.array(id_t)))

                    features = features1.cpu().detach().numpy()
                    features = np.concatenate((features, features2.cpu().detach().numpy()))

                else:
                    domains = np.concatenate((domains, domain1))
                    domains = np.concatenate((domains, domain2))

                    labels = np.concatenate((labels, np.array(id_s)))
                    labels = np.concatenate((labels, np.array(id_t)))

                    features = np.concatenate((features, features1.cpu().detach().numpy()))
                    features = np.concatenate((features, features2.cpu().detach().numpy()))

        return features, labels, domains


    def getdata(target, args):
        dataset = torch.utils.data.DataLoader(DATA(args=args, type=target, mode='test'),
                                              batch_size=1000,
                                              num_workers=args.workers,
                                              shuffle=True)
        return dataset


    m_name = 'mnistm'
    s_name = 'svhn'
    dataset1 = getdata(m_name, args)
    dataset2 = getdata(s_name, args)
    dataset = zip(dataset1, dataset2)
    #model_state = os.path.join(args.save_dir, 'model_DANN_mnistm-svhn.pth.tar')
    model_state = os.path.join(args.save_dir, 'model_DANN_svhn-mnistm.pth.tar')
    my_model = CNNModel().eval().cuda()
    my_model.load_state_dict(torch.load(model_state))
    data, label, domain = arrange_data(dataset, my_model)

    print("done")

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(data)

    df = pd.DataFrame()
    df['tsne-2d-one'] = tsne_results[:, 0]
    df['tsne-2d-two'] = tsne_results[:, 1]
    df['y'] = label

    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("hls", 10),
        data=df,
        legend="full",
        alpha=0.8
    )
    plt.show()

    df['y'] = domain

    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("hls", 2),
        data=df,
        legend="full",
        alpha=0.8
    )
    plt.show()
import torch.utils.data
from UDA.data_UDA import DATA


def test(dataset_name, epoch, args, my_model):


    """load data"""

    dataset = torch.utils.data.DataLoader(DATA(args=args, type=dataset_name, mode='test'),
                                         batch_size=args.test_batch,
                                         num_workers=args.workers,
                                         shuffle=True)

    """ training """
    my_net = my_model
    my_net = my_net.eval()
    my_net = my_net.cuda()

    len_dataloader = len(dataset)

    n_total = 0
    n_correct = 0
    alpha = 0

    for i, (images_s, id_s) in enumerate(dataset):

        images_s = images_s.cuda()
        id_s = id_s.cuda()
        batch_size = images_s.size(0)

        _, class_output = my_net(images_s)
        pred = class_output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(id_s.data.view_as(pred)).cpu().sum()
        n_total += batch_size

        i += 1

    accu = n_correct.data.numpy() * 1.0 / n_total

    print('epoch: %d, accuracy of the %s dataset: %f' % (epoch, dataset_name, accu))
    return accu
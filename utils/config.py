#encoding:utf-8
class DefaultConfig(object):
    root = './'
    data_path = './data_path.txt'

    Epoch_num = 90

    opt_num = 2
    batch_size_for_train = 24

    test_batch_num = 256
    batch_size_for_test = 2
    test_split_time = 2



    cut_out = True
    n_holes = 10
    length = 96
    local_shuffle_pretrain = True


    opt = 'Adam'
    save_sample_path = './samples/'
    learning_rate = 1e-4
    save_acc_thres = 0.05
    save_loss_thres = 10
    pretrain_image_pth_path = './pretrained_model/mobilenetv3-large-1cd25616.pth'
    patch_Local_model_path = './model/shuffle/'
    patch_random_number=3

    save_shuffle_path = './model/shuffle/'

    shuffle_train = True
    shuffle_test = True

    switch = {
    'O': 'Orcathus',
    'G': 'GreenBit',
    "D":'DigitalPersona'
    }

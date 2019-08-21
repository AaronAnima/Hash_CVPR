import os, time, multiprocessing
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from config import flags
from data import get_dataset_train, get_dataset_eval
from models import get_dwG, get_dwD
import random
import argparse


def hash_loss(hash_real, hash_aug, weighs_final, lambda_minEntrpBit):
    # consistent loss
    loss_consist = tl.cost.mean_squared_error(hash_aug, hash_real, is_mean=False)
    loss_consist = tf.cast(loss_consist, tf.float32)

    # independent loss
    t1 = tf.matmul(tf.transpose(weighs_final), weighs_final)
    t2 = tf.eye(weighs_final.shape[1])
    loss_independent = tl.cost.mean_squared_error(t1, t2, is_mean=False)
    loss_independent = tf.cast(loss_independent, tf.float32)

    # freq loss
    hash_freq = np.matmul(np.ones(flags.batch_size_train), hash_real) / (1.0 * flags.batch_size_train)

    loss_freq = tl.cost.sigmoid_cross_entropy(hash_freq, tf.ones_like(hash_freq)) +\
       tl.cost.sigmoid_cross_entropy(hash_freq, tf.zeros_like(hash_freq))
    loss_freq = tf.cast(loss_freq, tf.float32)

    # minimal entropy loss
    loss_min_entropy = tl.cost.sigmoid_cross_entropy(hash_real, tf.ones_like(hash_real)) +\
       tl.cost.sigmoid_cross_entropy(hash_real, tf.zeros_like(hash_real))
    loss_min_entropy = tf.cast(loss_min_entropy, tf.float32)

    # compute total loss
    total_loss = loss_consist + loss_freq - float(lambda_minEntrpBit) * loss_min_entropy + loss_independent

    return total_loss


def data_aug(images):

    z = np.random.normal(loc=0.0, scale=0.15,
        size = [flags.batch_size_train, flags.img_size_h, flags.img_size_h, flags.c_dim]).astype(np.float32)

    return images + z




def train():
    dataset, len_dataset = get_dataset_train()

    G = get_dwG([None, flags.z_dim], [None, flags.h_dim])
    D = get_dwD([None, flags.img_size_h, flags.img_size_w, flags.c_dim])

    G.train()
    D.train()

    n_step_epoch = int(len_dataset // flags.batch_size_train)

    lr_v = tf.Variable(flags.max_learning_rate)
    lr_decay = (flags.init_learning_rate - flags.max_learning_rate) / (n_step_epoch * flags.n_epoch)

    d_optimizer = tf.optimizers.Adam(lr_v, beta_1=flags.beta1)
    g_optimizer = tf.optimizers.Adam(lr_v, beta_1=flags.beta1)
    t_hash_loss_total = 10000

    for step, batch_imgs in enumerate(dataset):
        #print(batch_imgs.shape)
        lambda_minEntrpBit = flags.lambda_minEntrpBit
        lambda_Hash = flags.lambda_HashBit
        lambda_L2 = flags.lambda_L2

        # in first tenth epoch, no L2 & Hash loss
        if step//n_step_epoch == 0:
            lambda_L2 = 0
            lambda_Hash = 0

        # update learning rate
        lr_v.assign(lr_v + lr_decay)
        '''
        log = " ** new learning rate: %f (for GAN)" % (lr_v.tolist()[0])
        print(log)
        '''
        with tf.GradientTape(persistent=True) as tape:
            z = np.random.normal(loc=0.0, scale=1, size=[flags.batch_size_train, flags.z_dim]).astype(np.float32)
            # z = np.random.uniform(low=0.0, high=1.0, size=[flags.batch_size_train, flags.z_dim]).astype(np.float32)
            hash = np.random.randint(low=0.0, high=1.0, size=[flags.batch_size_train,flags.h_dim]).astype(np.float32)

            d_last_hidden_real, d_real_logits, hash_real = D(batch_imgs)
            fake_images = G([z, hash])
            d_last_hidden_fake, d_fake_logits, hash_fake = D(fake_images)

            weights_final = D.get_layer('hash_layer').all_weights[0]
            _, _, hash_aug= D(data_aug(batch_imgs))

            # adv loss
            d_loss_real = tl.cost.sigmoid_cross_entropy(d_real_logits, tf.ones_like(d_real_logits))
            d_loss_fake = tl.cost.sigmoid_cross_entropy(d_fake_logits, tf.zeros_like(d_fake_logits))
            g_loss_fake = tl.cost.sigmoid_cross_entropy(d_fake_logits, tf.ones_like(d_fake_logits))
            hash_l2_loss = tl.cost.mean_squared_error(hash, hash_fake, is_mean=True)

            # hash loss
            if t_hash_loss_total < flags.hash_loss_threshold:
                hash_loss_total = hash_loss(hash_real, hash_aug, weights_final, lambda_minEntrpBit[0])
            else:
                hash_loss_total = hash_loss(hash_real, hash_aug, weights_final, lambda_minEntrpBit[1])
            t_hash_loss_total = hash_loss_total # Save the new hash loss

            # feature matching loss (for generator)
            feature_matching_loss = tl.cost.mean_squared_error(d_last_hidden_real, d_last_hidden_fake, is_mean=True)

            # loss for discriminator
            d_loss = d_loss_real + d_loss_fake + \
                     lambda_L2 * hash_l2_loss + lambda_Hash * hash_loss_total
            g_loss = g_loss_fake
            # g_loss = g_loss_fake + feature_matching_loss

        grad = tape.gradient(d_loss, D.trainable_weights)
        d_optimizer.apply_gradients(zip(grad, D.trainable_weights))

        # grad = tape.gradient(feature_matching_loss, G.trainable_weights)
        grad = tape.gradient(g_loss, G.trainable_weights)
        g_optimizer.apply_gradients(zip(grad, G.trainable_weights))



        print("Epoch: [{}/{}] [{}/{}] L_D: {:.6f}, L_G: {:.6f}, L_Hash: {:.3f}, "
              "L_adv: {:.6f}, L_2: {:.3f}".format
              (step//n_step_epoch, flags.n_epoch, step, n_step_epoch, d_loss, g_loss,
               lambda_Hash * hash_loss_total, d_loss_real + d_loss_fake, lambda_L2 * hash_l2_loss))

        if np.mod(step, flags.save_step) == 0:
            G.save_weights('{}/G.npz'.format(flags.checkpoint_dir), format='npz')
            D.save_weights('{}/D.npz'.format(flags.checkpoint_dir), format='npz')
            G.train()
        if np.mod(step, flags.eval_step) == 0:
            z = np.random.normal(loc=0.0, scale=1, size=[flags.batch_size_train, flags.z_dim]).astype(np.float32)
            # z = np.random.uniform(low=0.0, high=1.0, size=[flags.batch_size_train, flags.z_dim]).astype(np.float32)
            hash = np.random.randint(low=0.0, high=1.0, size=[flags.batch_size_train, flags.h_dim]).astype(np.float32)
            G.eval()
            result = G([z, hash])
            G.train()
            tl.visualize.save_images(result.numpy(), [8, 8],
                                     '{}/train_{:02d}_{:04d}.png'.format(flags.sample_dir, step // n_step_epoch, step))
        del tape


class Retrival_Obj():
    def __init__(self, hash, label):
        self.label = label
        self.dist = 0
        list1 = [True if hash[i] == 1 else False for i in range(len(hash))]
        # convert bool list to bool array
        self.hash = np.array(list1)

    def __repr__(self):
        return repr((self.hash, self.label, self.dist))

# to calculate the hamming dist between obj1 & obj2


def hamming(obj1, obj2):
    res = obj1.hash ^ obj2.hash
    ans = 0
    for k in range(len(res)):
        if res[k] == True :
            ans += 1
    obj2.dist = ans


def take_ele(obj):
    return obj.dist


# to get 'nearest_num' nearest objs from 'image' in 'Gallery'
def get_nearest(image, Gallery, nearest_num):
    for obj in Gallery:
        hamming(image, obj)
    Gallery.sort(key=take_ele)
    ans = []
    cnt = 0
    for obj in Gallery:
        cnt += 1
        if cnt <= nearest_num:
            ans.append(obj)
        else:
            break

    return ans


# given retrivial_set, calc AP w.r.t. given label
def calc_ap(retrivial_set, label):
    total_num = 0
    ac_num = 0
    ans = 0
    result = []
    for obj in retrivial_set:
        total_num += 1
        if obj.label == label:
            ac_num += 1
        ans += ac_num / total_num
        result.append(ac_num / total_num)
    result = np.array(result)
    ans = np.mean(result)
    return ans


def Evaluate_mAP():
    print('Start Eval!')
    # load images & labels
    ds = get_dataset_eval()
    D = get_dwD([None, flags.img_size_h, flags.img_size_w, flags.c_dim])
    D.load_weights('./checkpoint/D.npz')
    D.eval()

    # create (hash,label) gallery
    Gallery = []
    cnt = 0
    step_time1 = time.time()
    for batch, label in ds:
        cnt += 1
        if cnt % flags.eval_print_freq == 0:
            step_time2 = time.time()
            print("Now {} Imgs done, takes {:.3f} sec".format(cnt, step_time2 - step_time1))
            step_time1 = time.time()
        _, _, hash_fake = D(batch)
        hash_fake = hash_fake.numpy()[0]
        hash_fake = ((tf.sign(hash_fake*2 -1, name=None) + 1)/2).numpy()
        label = label.numpy()[0]
        Gallery.append(Retrival_Obj(hash_fake, label))
    print('Hash calc done, start split dataset')

    #sample 1000 from Gallery and bulid the Query set
    random.shuffle(Gallery)
    cnt = 0
    Queryset = []
    G = []
    for obj in Gallery:
        cnt += 1
        if cnt > flags.eval_sample:
            G.append(obj)
        else:
            Queryset.append(obj)
    Gallery = G
    print('split done, start eval')

    # Calculate mAP
    Final_mAP = 0
    step_time1 = time.time()
    for eval_epoch in range(flags.eval_epoch_num):
        result_list = []
        cnt = 0
        for obj in Queryset:
            cnt += 1
            if cnt % flags.retrieval_print_freq == 0:
                step_time2 = time.time()
                print("Now Steps {} done, takes {:.3f} sec".format(eval_epoch, cnt, step_time2 - step_time1))
                step_time1 = time.time()

            retrivial_set = get_nearest(obj, Gallery, flags.nearest_num)
            result = calc_ap(retrivial_set, obj.label)
            result_list.append(result)
        result_list = np.array(result_list)
        temp_res = np.mean(result_list)
        print("Query_num:{}, Eval_step:{}, Top_k_num:{}, AP:{:.3f}".format(flags.eval_sample, eval_epoch,
                                                                           flags.nearest_num, temp_res))
        Final_mAP += temp_res / flags.eval_epoch_num
    print('')
    print("Query_num:{}, Eval_num:{}, Top_k_num:{}, mAP:{:.3f}".format(flags.eval_sample, flags.eval_epoch_num,
                                                                    flags.nearest_num, Final_mAP))
    print('')


def Evaluate_Cluster():
    return 0


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='DWGAN', help='DWGAN, evaluate')
    args = parser.parse_args()

    if args.mode == 'DWGAN':
        train()
    elif args.mode == 'eval_mAP':
        Evaluate_mAP()
    elif args.mode == 'eval_clustering':
        Evaluate_Cluster()
    elif args.mode == 'both' :
        train()
        Evaluate_mAP()
    else:
        raise Exception("Unknow --mode")

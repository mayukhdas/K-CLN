import prepare_data
import gzip
import cPickle
import numpy
import read_pkl as rp
import random




def sample_data(path, portion):
    f = gzip.open(path, 'rb')
    feats, labels, rel_list, train_ids, valid_ids, test_ids = cPickle.load(f)
    feats, labels = prepare_data.modData(feats, labels)

    train_ids_sample = random.sample(train_ids, int(len(train_ids)*portion))
    valid_ids_sample = random.sample(valid_ids, int(len(valid_ids)*portion))
    test_ids_sample = random.sample(test_ids, int(len(test_ids)*portion))
    # index = list()
    # index.extend(train_ids_sample)
    # index.extend(valid_ids_sample)
    # index.extend(test_ids_sample)
    # feats_sample = feats[index]
    # labels_sample = labels[index]
    # rel_list_sample = list()
    # for rel in [rel_list[i] for i in index]:
    #
    #     rel_item_0 = list()
    #     for item_index in rel[0]:
    #         if item_index in index:
    #             rel_item_0.append(item_index)
    #     rel_item_1 = list()
    #     for item_index in rel[1]:
    #         if item_index in index:
    #             rel_item_1.append(item_index)
    #     rel_item_tuple = list()
    #     rel_item_tuple.append(rel_item_0)
    #     rel_item_tuple.append(rel_item_1)
    #     rel_list_sample.append(rel_item_tuple)
    # rel_list_sample, rel_mask_sample = prepare_data.create_mask(rel_list_sample)
    rel_list, rel_mask = prepare_data.create_mask(rel_list)

    # return feats_sample, labels_sample, rel_list_sample, rel_mask_sample, train_ids_sample, valid_ids_sample, test_ids_sample
    return feats, labels, rel_list, rel_mask, train_ids_sample, valid_ids_sample, test_ids_sample




# feats_sample, labels_sample, rel_list_sample, rel_mask_sample, train_ids_sample, valid_ids_sample, test_ids_sample = sample_data('../data/' + 'pubmed' + '.pkl.gz', 0.01)
# print feats_sample.shape, labels_sample.shape
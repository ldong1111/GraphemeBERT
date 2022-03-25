# -*- coding: utf-8 -*-
#! / usr/bin/env python3
# python2的print bug
from __future__ import print_function


import torch
import torch.nn as nn
import torch.optim as optim

import torchtext

import torch.nn.functional as F


from torchtext.data import Field, BucketIterator
from torchtext import data, datasets


import numpy as np

import random
import math
import time

import copy

from argparse import ArgumentParser

from collections import Counter


parser = ArgumentParser(description='GBERT')


parser.add_argument('--pretrain_language',
                    type=str,
                    default='dut')

parser.add_argument('--mask_ratio',
                    type=float,
                    default=0.2)



parser.add_argument('--debug', action='store_true') 


config = parser.parse_args()

print('\nconfig')
print(config)

# pretrain language
pretrain_language = config.pretrain_language

# debug flag
global_test_flag = config.debug


def tokenize_grapheme_only_letter_part(enhanced_word):
    """
    tokenizer word without langid (For monolingual)
    """
    index_over = enhanced_word.index('}')
    lang_id = [enhanced_word[:index_over + 1]]

    splited_word = [enhanced_word[i] for i in range(index_over + 1, len(enhanced_word))]

    return splited_word


def tokenize_grapheme_langid(enhanced_word):
    """
    tokenizer the langid  of word (For multilingual GBERT)
    """
    index_over = enhanced_word.index('}')
    lang_id = [enhanced_word[:index_over + 1]]

    return lang_id



SRC = Field(sequential=True, tokenize=tokenize_grapheme_only_letter_part,
            eos_token='<eos>', init_token='<sos>',
            lower=True, batch_first=True)


LANGID = Field(sequential=False, use_vocab=False)


fields_pretrain = [('merged_spelling', SRC), ('langid', LANGID)]

# this is used for counting the mask accuracy for different languages (multilingual case)
langid_with_brace = '{' + pretrain_language + '}'
whole_language_code_dict_reverse_np = {0: langid_with_brace}

print('\nlanguage list')
print(whole_language_code_dict_reverse_np)


train_path = pretrain_language + '_word_data_train_without_g2p_dev_and_test_word.csv'
validate_path = pretrain_language + '_word_data_validate_without_g2p_dev_and_test_word.csv'

dir_root_path = '.'

word_dictionary_path = dir_root_path +  '/monolingual_medium_resource/monolingual_word_data'

train_data, valid_data = data.TabularDataset.splits(
    path=word_dictionary_path, format='csv',
    train=train_path, validation=validate_path, skip_header=True, fields=fields_pretrain)

# build vocab of training set
min_freq = 5

SRC.build_vocab(train_data, min_freq=min_freq)


src_vocab_list = SRC.vocab.itos
src_vocab_dict = SRC.vocab.stoi

i = 0
for x in src_vocab_dict:
    print(x, end='\t')
    i += 1
    if i >= 10:
        break

print(src_vocab_dict)

if '#' in src_vocab_dict:
    print('\nError, ')
    exit(0)

print('\nSRC vocab (Add a mask token #)c-------------------------------------')

src_vocab_list.extend('#')

# change the original vocab 
# Original: unk 0, pad 1, sos 2, eos 3, Others 4+
# New(Added a mask(# 4): unk0, pad1, sos2, eos3, [Mask (#)]4, Others 5+

for x in src_vocab_dict:
    if src_vocab_dict[x] >= 4:  
        src_vocab_dict[x] = src_vocab_dict[x] + 1  

src_vocab_dict['#'] = 4 

print('\nThe training vocab(Added a mask token #)')
print(src_vocab_dict)


# save the torchtext vocab
src_vocab = SRC.vocab
save_path =  dir_root_path + '/monolingual_medium_resource' +  '/pretrain_model_vocab/' + pretrain_language + '_wikipron_pretrained_random_mask_lm_vocab.pth'

torch.save(src_vocab, save_path)



src_vocab_extended = SRC.vocab.itos
src_vocab_dict_extended = SRC.vocab.stoi

# get the reverse vocab (0: unk, 1: pad, ...)
src_vocab_dict_extended_reversed = {}
for _, key in enumerate(src_vocab_dict_extended):
    src_vocab_dict_extended_reversed[src_vocab_dict_extended[key]] = key


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


BATCH_SIZE = 256  # accumulated training for 4 times

EQUAL_BATCH_SIZE = 1024

ACCUMULATE_NUMBER = EQUAL_BATCH_SIZE // BATCH_SIZE

RELU_FLAG = False  # relu/gelu, the default is gelu following BERT.

LABEL_SMOOTHED_LOSS_FLAG = True

SMOOTHING = 0.1

train_iterator, valid_iterator = BucketIterator.splits(
    (train_data, valid_data),
    batch_size=BATCH_SIZE,
    sort=False,
    device=device)

# The GBERT (a Transformer encoder)
class MaskedLMEncoder(nn.Module):
    def __init__(self,
                 input_dim,
                 langid_dim,
                 src_pad_idx,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 max_length=100):
        super().__init__()

        self.device = device

        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)


        self.src_pad_idx = src_pad_idx                                                                                        # device)

        self.layers = nn.ModuleList([EncoderLayer(hid_dim,
                                                 n_heads,
                                                 pf_dim,
                                                 dropout,
                                                 device)
                                     for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

        # the grapheme prediction FC, which will be discarded during downstream G2P task
        self.grapheme_prediction_layer = nn.Linear(hid_dim, input_dim)


        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def make_src_mask(self, src):

        # src = [batch size, src len]

        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        # src_mask = [batch size, 1, 1, src len]

        return src_mask

    def forward(self, src, src_lang):

        # src = [batch size, src len]
        # src_mask = [batch size, src len]

        batch_size = src.shape[0]
        src_len = src.shape[1]

        # src_mask

        src_mask = self.make_src_mask(src)


        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        src = self.dropout((( self.tok_embedding(src) )  * self.scale) + self.pos_embedding(pos))


        for layer in self.layers:

            src = layer(src, src_mask)

        src_grapheme_prediction = self.grapheme_prediction_layer(src)


        argmax_output = src_grapheme_prediction.argmax(-1)

        # return: contextual grapheme representations, grapheme_prediction_logits, grapheme_predictions
        return src, src_grapheme_prediction, argmax_output

# Transformer encoder layer
class EncoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()

        self.layer_norm = nn.LayerNorm(hid_dim)

        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)

        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):

        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, src len]

        #self attention

        _src, _ = self.self_attention(src, src, src, src_mask)

        #dropout, residual connection and layer norm
        src = self.layer_norm(src + self.dropout(_src))

        #src = [batch size, src len, hid dim]

        _src = self.positionwise_feedforward(src)

        #dropout, residual and layer norm
        src = self.layer_norm(src + self.dropout(_src))

        #src = [batch size, src len, hid dim]

        return src

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads


        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)


        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # Q = [batch size, query len, hid dim]
        # K = [batch size, key len, hid dim]
        # V = [batch size, value len, hid dim]


        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Q = [batch size, n heads, query len, head dim]
        # K = [batch size, n heads, key len, head dim]
        # V = [batch size, n heads, value len, head dim]

        # Q = [batch size, n heads, query len, head dim]
        # K = [batch size, n heads，head dim， key len ]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy = [batch size, n heads, query len, key len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)

        # attention = [batch size, n heads, query len, key len]


        x = torch.matmul(self.dropout(attention), V)

        # x = [batch size, n heads, query len, head dim]


        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim)

        # x = [batch size, query len, hid dim]

        x = self.fc_o(x)

        # x = [batch size, query len, hid dim]

        return x, attention


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        # x = [batch size, seq len, hid dim]
        if not RELU_FLAG:
            x = self.dropout(gelu(self.fc_1(x)))
        else:
            x = self.dropout(torch.relu(self.fc_1(x)))
        # x = [batch size, seq len, pf dim]

        x = self.fc_2(x)

        # x = [batch size, seq len, hid dim]

        return x


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


INPUT_DIM = len(SRC.vocab)
LANGID_DIM = len(whole_language_code_dict_reverse_np)

HID_DIM = 256
ENC_LAYERS = 6  # 6-layer Transformer encoder (GBERT)

ENC_HEADS = 4

ENC_PF_DIM = 1024

ENC_DROPOUT = 0.1

# The pad index is 1
SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]


TRG_PAD_IDX = 1


enc = MaskedLMEncoder(INPUT_DIM,
                      LANGID_DIM,
                      SRC_PAD_IDX,
                      HID_DIM,
                      ENC_LAYERS,
                      ENC_HEADS,
                      ENC_PF_DIM,
                      ENC_DROPOUT,
                      device)

TRG_PAD_IDX = 1


model = enc.to(device)
# import the initial parameters of all LSTM and FC.
model.apply(initialize_weights)

print('model parameters')
print(model)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')

criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

# The eos index is 3
TRG_EOS_IDX = 3

# label smoothing loss
def label_smoothed_nll_loss(prediction, target, epsilon, ignore_index=None, print_flag=False, reduce=True):

    lprobs = torch.log(torch.softmax(prediction, dim=-1))


    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)

    nll_loss = -lprobs.gather(dim=-1, index=target)

    nll_loss_original = -lprobs.gather(dim=-1, index=target)
    nll_loss_original_ave = torch.mean(nll_loss_original)


    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)


    if True:  # ignore_index is not None:
        pad_mask = target.eq(ignore_index)

        non_ignored_num = torch.sum((pad_mask == False))

        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)


    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()

    nll_loss_ave = nll_loss / non_ignored_num
    smooth_loss_ave = smooth_loss / non_ignored_num

    eps_i = epsilon / lprobs.size(-1)
    loss_ave = (1 - epsilon) * nll_loss_ave + eps_i * smooth_loss_ave

    return loss_ave, nll_loss_ave


def get_random_k_index(sequence_length, mask_length, print_flag=False):
    '''
    get the random k index from [1, seq len], since the input format is [<sos> grapheme_1 grapheme_2 ... grapheme_n <eos>]
    the indexes of <sos> and <eos> are not included.
    '''
    # the index interval is [1, seq len] 
    index_list = list(range(1, 1 + sequence_length))

    # shuffle
    random.shuffle(index_list)

    selected_index_list = sorted(index_list[:mask_length])

    return selected_index_list


# generated the masked input and mask ground truth
def generate_source_random_mask(x, device, x_unspecial_start_index, x_vocab_end_index, mask_ratio=0.2, print_flag=False,
                         mask_index=4, sos_index=2, eos_index=3, pad_index=1):
    '''
    input: x [batch size, src len]
    output: masked_input and mask ground truth
    :param x_unspecial_start_index:
    :param mask_ratio:
    :param x:
    :param device:
    :return:
    '''

    batch_size = x.shape[0]
    trg_len = x.shape[1]

    # the results to save: unmasked_input, output, unmasked_input_for_test(without 8:1:1 random/original/mask disturbance )
    x_unmasked = torch.zeros(batch_size, trg_len).long().to(device)
    output = torch.zeros(batch_size, trg_len).long().to(device)
    x_unmasked_for_test = torch.zeros(batch_size, trg_len).long().to(device)

    # the default value is <pad>
    x_unmasked[:, :] = pad_index
    output[:, :] = pad_index

    x_unmasked_for_test[:, :] = pad_index

    selected_batch = random.randint(0, batch_size - 1)


    for i in range(batch_size):

        # get the eos position
        this_eos_position = torch.where(x[i] == eos_index)[0]


        this_length = this_eos_position.item() - 1

        this_print_flag = False
        if print_flag and i == selected_batch:
            this_print_flag = True

            print('\ncurrent x')
            print(x[i])
            print('eos_index and sentence length')
            print(this_eos_position)
            print(this_length)

        # get the mask length 
        this_mask_length = max(1, round(this_length * mask_ratio))

        # get random masked index
        selected_index_list = get_random_k_index(this_length, this_mask_length, this_print_flag)

        # set the <sos> and <eos> values
        x_unmasked[i, 0] = sos_index
        x_unmasked[i, this_eos_position] = eos_index

        x_unmasked_for_test[i, 0] = sos_index
        x_unmasked_for_test[i, this_eos_position] = eos_index


        for j in range(1, this_eos_position):
            if j in selected_index_list:  # done for the masked index list
                output[i, j] = x[i, j]

                x_unmasked_for_test[i, j] = mask_index

                # 0.8/0.1/0.1 disturbance
                this_random_number = random.random()
                if this_random_number < 0.8:
                    x_unmasked[i, j] = mask_index
                elif this_random_number < 0.9:
                    x_unmasked[i, j] = random.randint(x_unspecial_start_index, x_vocab_end_index)
                else:
                    x_unmasked[i, j] = x[i, j]

            else:
                x_unmasked[i, j] = x[i, j]

                x_unmasked_for_test[i, j] = x[i, j]
        if print_flag and i == selected_batch:
            print('\nx_maske range' )
            print(selected_index_list)

            print('\ncurrent x')
            print(x[i])
            print('\nx_unmasked')
            print(x_unmasked_for_test[i])
            print('\noutput（ground truth)')
            print(output[i])

            print('\nx_unmasked（include  8:1:1 mask/random(unspecial--end）/word')
            print(x_unmasked[i])

    output = output.clone().detach()

    x_unmasked = x_unmasked.clone().detach()


    return x_unmasked, output


def compute_masked_lm_acc(prediction, ground_truth, print_flag=False, pad_index=1):
    '''
    input: [batch size, trg len]

    
    '''
    batch_size = ground_truth.shape[0]

    mask_length_list = []

    this_batch_mask_number = 0
    this_batch_mask_prediction_right_number = 0

    selected_batch_index = random.randint(0, batch_size - 1)

    for i in range(batch_size):
        this_mask_index_list = torch.where(ground_truth[i] != pad_index)[0]
        this_mask_length = this_mask_index_list.shape[0]

  
        mask_length_list.append(this_mask_length)

        this_batch_mask_number += this_mask_length

        this_ground_truth = ground_truth[i, this_mask_index_list]
        this_prediction = prediction[i, this_mask_index_list]

        # the right prediction number of this batch
        this_mask_prediction_right_number = (this_ground_truth == this_prediction).sum().item()

        this_batch_mask_prediction_right_number += this_mask_prediction_right_number

    # return mask length list, whole right prediction number, whole masked number
    return this_batch_mask_prediction_right_number, this_batch_mask_number, mask_length_list


def compute_masked_lm_acc_for_each_language(prediction, ground_truth, src_langid, divided_languages_mask_acc_message, print_flag=False, pad_index=1):
    '''
    input: [batch size, trg len]

    compute the mask acc for different language
    '''
    batch_size = ground_truth.shape[0]

    mask_length_list = []

    this_batch_mask_number = 0
    this_batch_mask_prediction_right_number = 0

    selected_batch_index = random.randint(0, batch_size - 1)

    for i in range(batch_size):

        # current language
        this_language = src_langid[i]

        this_mask_index_list = torch.where(ground_truth[i] != pad_index)[0]
        this_mask_length = this_mask_index_list.shape[0]


        mask_length_list.append(this_mask_length)


        this_batch_mask_number += this_mask_length

        this_ground_truth = ground_truth[i, this_mask_index_list]
        this_prediction = prediction[i, this_mask_index_list]

        this_mask_prediction_right_number = (this_ground_truth == this_prediction).sum().item()

        this_batch_mask_prediction_right_number += this_mask_prediction_right_number

        # update the message of different language
        divided_languages_mask_acc_message[this_language]['whole_mask_number'] += this_mask_length
        divided_languages_mask_acc_message[this_language]['whole_mask_prediction_right_number'] += this_mask_prediction_right_number
        divided_languages_mask_acc_message[this_language]['mask_len_list'].append(this_mask_length)

    return divided_languages_mask_acc_message

# pretraining
def pretrain_train(model, iterator, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0

    optimizer.zero_grad()  # accumulate training

    x_unspecial_start_index = 4  # unk0, pad1, sos2, eos3, mask(#) 4
    x_vocab_end_index = INPUT_DIM - 1  

    selected_batch_index = random.randint(0, 10)

    # information for computing mask acc
    whole_mask_len_list = []

    whole_mask_prediction_right_number = 0

    whole_mask_number = 0

    for i, batch in enumerate(iterator):

        src = batch.merged_spelling



        src_langid = batch.langid


        if i == 0:
            print('\n------------------------pretrain train--------------------------------')

        print_flag = False
        if i == selected_batch_index:
            print_flag = True

            batch_size = src.shape[0]
            this_random_index = random.randint(0, batch_size - 1)

            this_src = src[this_random_index].contiguous().data.cpu().numpy()
            this_src_langid = src_langid[this_random_index].contiguous().data.cpu().numpy()

            print('\nsrc')
            print(this_src)

            print('src_lang')
            print(this_src_langid)

            this_src_reverse = []
            this_src_langid_reverse = []

            pad_index = 1 
            this_src_non_pad_position = np.where(this_src != pad_index)[0]
            for item in this_src[this_src_non_pad_position]:
                this_src_reverse.append(src_vocab_dict_extended_reversed[item])
            this_src_langid_reverse = whole_language_code_dict_reverse_np[int(this_src_langid)]

            print('\nsrc（reverse)')
            print(this_src_reverse)
            print(''.join(this_src_reverse))
            print('src_lang')
            print(this_src_langid_reverse)

        src_len = src.shape[1]

        src_langid = src_langid.unsqueeze(1).repeat(1, src_len)

        src_unmasked, src_masked_ground_truth = generate_source_random_mask(src, device, x_unspecial_start_index,
                                                                     x_vocab_end_index, config.mask_ratio,
                                                                     print_flag)

        # the inputs are src, langid
        src_encoder_representation, output, argmax_output = model(src_unmasked, src_langid)

        # update the mask acc message
        this_batch_mask_prediction_right_number, this_batch_mask_number, this_mask_length_list = \
            compute_masked_lm_acc(argmax_output, src_masked_ground_truth, print_flag, pad_index=1)

        whole_mask_len_list.extend(this_mask_length_list)
        whole_mask_prediction_right_number += this_batch_mask_prediction_right_number
        whole_mask_number += this_batch_mask_number

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        src_masked_ground_truth = src_masked_ground_truth.contiguous().view(-1)


        if not LABEL_SMOOTHED_LOSS_FLAG:
            loss = criterion(output, src_masked_ground_truth)
        else:
            PAD_INDEX = 1  # pad_index is 1
            loss, _ = label_smoothed_nll_loss(output, src_masked_ground_truth, SMOOTHING, ignore_index=PAD_INDEX,
                                              print_flag=print_flag)

        loss = loss / ACCUMULATE_NUMBER  # accumulate training

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        if (i + 1) % ACCUMULATE_NUMBER == 0 or ((i + 1) == len(iterator)): 
           
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss += loss.item()

        if global_test_flag and i == 10:  # debug
            break

    # get the mask length distribution
    whole_mask_len_list_counter = Counter(whole_mask_len_list)

    mask_acc = whole_mask_prediction_right_number / whole_mask_number
    print('\nmask acc of training set %.4f = %d/%d' % (mask_acc, whole_mask_prediction_right_number, whole_mask_number))
    print('\nthe mask length distribution of training set')
    print(whole_mask_len_list_counter.most_common())

    return epoch_loss / len(iterator), mask_acc



def pretrain_evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    x_unspecial_start_index = 4  # unk0, pad1, sos2, eos3, mask(#) 4
    x_vocab_end_index = INPUT_DIM - 1  
    
    selected_batch_index = random.randint(0, 10)

    # information for computing mask acc
    whole_mask_len_list = []

    whole_mask_prediction_right_number = 0

    whole_mask_number = 0

    # whole_mask_number, whole_mask_prediction_number, mask_len_list
    divided_languages_mask_acc_message = {}

    for i_index in whole_language_code_dict_reverse_np: 
        # 
        this_language = whole_language_code_dict_reverse_np[i_index][1:-1]
        divided_languages_mask_acc_message[this_language] = {}
        divided_languages_mask_acc_message[this_language]['whole_mask_number'] = 0
        divided_languages_mask_acc_message[this_language]['whole_mask_prediction_right_number'] = 0
        divided_languages_mask_acc_message[this_language]['mask_len_list'] = []

    with torch.no_grad():
        for i, batch in enumerate(iterator):

            if i == 0:
                print('\n-----------pretrain validate-----------------------')
            src = batch.merged_spelling
            # print(src)

            # print(trg)

            src_langid = batch.langid


            src_langid_np = src_langid.contiguous().data.cpu().numpy() 

            src_langid_string_list = []
            for item in src_langid_np:

                src_langid_string_list.append(whole_language_code_dict_reverse_np[int(item)][1:-1])


            print_flag = False
            if i == selected_batch_index:
                print_flag = True

                batch_size = src.shape[0]
                this_random_index = random.randint(0, batch_size - 1)

                this_src = src[this_random_index].contiguous().data.cpu().numpy()
                this_src_langid = src_langid[this_random_index].contiguous().data.cpu().numpy()

                print('\nsrc')
                print(this_src)

                print('src_lang')
                print(this_src_langid)

                this_src_reverse = []
                this_src_langid_reverse = []

                pad_index = 1 
                this_src_non_pad_position = np.where(this_src != pad_index)[0]
                for item in this_src[this_src_non_pad_position]:
                    this_src_reverse.append(src_vocab_dict_extended_reversed[item])

                this_src_langid_reverse = whole_language_code_dict_reverse_np[int(this_src_langid)]

                print('\nsrc（reverse)')
                print(this_src_reverse)
                print(''.join(this_src_reverse))
                print('src_lang')
                print(this_src_langid_reverse)

            src_len = src.shape[1]

            src_langid = src_langid.unsqueeze(1).repeat(1, src_len)

            src_unmasked, src_masked_ground_truth = generate_source_random_mask(src, device, x_unspecial_start_index,
                                                                         x_vocab_end_index, config.mask_ratio,
                                                                         print_flag)


            src_encoder_representation, output, argmax_output = model(src_unmasked, src_langid)

            # update the information of mask number, mask length
            this_batch_mask_prediction_right_number, this_batch_mask_number, this_mask_length_list = \
                compute_masked_lm_acc(argmax_output, src_masked_ground_truth, print_flag, pad_index=1)

            # update the information of each language
            divided_languages_mask_acc_message = compute_masked_lm_acc_for_each_language(argmax_output, src_masked_ground_truth,src_langid_string_list, divided_languages_mask_acc_message, print_flag, pad_index=1)

            whole_mask_len_list.extend(this_mask_length_list)
            whole_mask_prediction_right_number += this_batch_mask_prediction_right_number
            whole_mask_number += this_batch_mask_number

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            src_masked_ground_truth = src_masked_ground_truth.contiguous().view(-1)


            if not LABEL_SMOOTHED_LOSS_FLAG:
                loss = criterion(output, src_masked_ground_truth)
            else:
                PAD_INDEX = 1  
                loss, _ = label_smoothed_nll_loss(output, src_masked_ground_truth, SMOOTHING, ignore_index=PAD_INDEX,
                                                  print_flag=print_flag)

            loss = loss / ACCUMULATE_NUMBER 

            epoch_loss += loss.item()

            if global_test_flag and i == 10:  # debug
                break

    whole_mask_len_list_counter = Counter(whole_mask_len_list)

    mask_acc = whole_mask_prediction_right_number / whole_mask_number
    print('\nmask acc of the valudation set: %.4f = %d/%d' % (mask_acc, whole_mask_prediction_right_number, whole_mask_number))
    print('\nthe mask length distribution of the validation set')
    print(whole_mask_len_list_counter.most_common())

    mask_acc_list = []
    mask_acc_dict = {}
    for item in divided_languages_mask_acc_message:
        divided_languages_mask_acc_message[item]['mask_acc'] = divided_languages_mask_acc_message[item]['whole_mask_prediction_right_number'] / divided_languages_mask_acc_message[item]['whole_mask_number']
        print('%s language, mask acc: %.4f = %d/%d' % (item, divided_languages_mask_acc_message[item]['mask_acc'], divided_languages_mask_acc_message[item]['whole_mask_prediction_right_number'], divided_languages_mask_acc_message[item]['whole_mask_number']))
        print('mask_length distribution')
        print(Counter(divided_languages_mask_acc_message[item]['mask_len_list']).most_common())

        this_mask_acc = round(divided_languages_mask_acc_message[item]['mask_acc'], 4) 
        mask_acc_list.append(this_mask_acc)

        mask_acc_dict[item] = this_mask_acc

    print('\nmask_acc_list')
    print(mask_acc_list)
    print('\nmask_acc_dict')
    print(mask_acc_dict)

    return epoch_loss / len(iterator), mask_acc



def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# training loss
Train_loss_list = []
Valid_loss_list = []
Train_wer_list = []
Valid_wer_list = []
Train_per_list = []
Valid_per_list = []


train_mask_acc_list = []
valid_mask_acc_list = []

N_EPOCHS = 400 

if config.debug:
    N_EPOCHS = 5
CLIP = 1

LR = 0.0001

WARMUP_INIT_LR = 1e-7
WARMUP_UPDATES = int(0.1 * N_EPOCHS)

INVERSE_SQER_LR_FLAG = True

WEIGHT_DECAY = 0  

SAVE_PATH = dir_root_path + '/monolingual_medium_resource' + '/torch_models/g2p-model_' + pretrain_language + '_without_dev_and_test_g2p_data_random_mask_valid_mask_loss.pt'

SAVE_PATH_BEST_VALID_MASK_ACC = dir_root_path + '/monolingual_medium_resource' + '/torch_models/g2p-model_' + pretrain_language + '_without_dev_test_g2p_data_random_mask_valid_mask_acc.pt'

best_valid_loss = float('inf')

best_valid_mask_acc = -1

print('config')
print(config)

print('\nBATCH_SIZE : %d, accumulated batch size=%d %d' % (EQUAL_BATCH_SIZE, BATCH_SIZE, ACCUMULATE_NUMBER))
print('EPOCHS: %d, LR=%.5f' % (N_EPOCHS, LR))
print('Embedding hid dim = %d, pf_dim = %d, layers: %d, multi head number: %d' % (HID_DIM, ENC_PF_DIM, ENC_LAYERS, ENC_HEADS))
print('\nactivation function: relu_flag: ', RELU_FLAG)

print('\nlabel_smoothing_loss: ', LABEL_SMOOTHED_LOSS_FLAG)
print('smoothing: %.2f' % SMOOTHING)

train_data_number = len(train_data.examples)

if INVERSE_SQER_LR_FLAG:
    print('\nwarmup_lr: %.8f %d epochs warmup (%d batch size), %d epochs sqrt' % (
        WARMUP_INIT_LR, WARMUP_UPDATES, WARMUP_UPDATES * train_data_number / BATCH_SIZE, N_EPOCHS))
else:
    print('\nlr: %.5f' % LR)

print('\nAdam weight dacay: %.4f' % WEIGHT_DECAY)


lrs = torch.linspace(WARMUP_INIT_LR, LR, WARMUP_UPDATES)

# training
for epoch in range(N_EPOCHS):

    # print(lrs)

    if INVERSE_SQER_LR_FLAG:
        if epoch < WARMUP_UPDATES:
            lr = lrs[epoch]
        else:
            decay_factor = (WARMUP_UPDATES / (epoch + 1)) ** 0.5
            lr = LR * decay_factor

    else:
        lr = LR

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    start_time = time.time()

    train_loss, train_mask_acc = pretrain_train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss, valid_mask_acc = pretrain_evaluate(model, valid_iterator, criterion)

    Train_loss_list.append(train_loss)

    Valid_loss_list.append(valid_loss)

    train_mask_acc_list.append(train_mask_acc)
    valid_mask_acc_list.append(valid_mask_acc)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss

        torch.save(model.state_dict(), SAVE_PATH)

    # the checkpoint with the best valid mask acc will be saved
    if valid_mask_acc > best_valid_mask_acc:
        best_valid_mask_acc = valid_mask_acc

        torch.save(model.state_dict(), SAVE_PATH_BEST_VALID_MASK_ACC)

    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.4f} | Val. Loss: {valid_loss:.4f}')
    print(f'\tTrain mask acc: {train_mask_acc:.4f} | Val. mask acc: {valid_mask_acc:.4f}')

    print('\nCurrent train loss list')
    print([round(x, 4) for x in Train_loss_list])
    print('\nCurrent valid loss list')
    print([round(x, 4) for x in Valid_loss_list])

    print('\nTrain mask acc list')
    print([round(x, 4) for x in train_mask_acc_list])

    print('\nValid mask acc list')
    print([round(x, 4) for x in valid_mask_acc_list])


# test for final GBERT model
model.load_state_dict(torch.load(SAVE_PATH))

valid_loss, valid_mask_acc = pretrain_evaluate(model, valid_iterator, criterion)

model.load_state_dict(torch.load(SAVE_PATH_BEST_VALID_MASK_ACC))

valid_loss, valid_mask_acc = pretrain_evaluate(model, valid_iterator, criterion)

print('\nTrain loss list')
print([round(x, 4) for x in Train_loss_list])
print('\nValid loss list')
print([round(x, 4) for x in Valid_loss_list])
print('\nTrain mask acc list')
print([round(x, 4) for x in train_mask_acc_list])

print('\nValid mask acc list')
print([round(x, 4) for x in valid_mask_acc_list])


best_valid_mask_acc_epoch = valid_mask_acc_list.index(max(valid_mask_acc_list))
print('\nfinal result')
print('\nbest valid mask acc: %d/%d epoch' % (best_valid_mask_acc_epoch, N_EPOCHS))
print('\nfinal train mask acc: %.4f, valid mask acc: %.4f' % (
    train_mask_acc_list[best_valid_mask_acc_epoch], valid_mask_acc_list[best_valid_mask_acc_epoch]))





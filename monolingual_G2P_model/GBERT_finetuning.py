import torch
import torch.nn as nn
import torch.optim as optim

import torchtext

import torch.nn.functional as F

from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator
from torchtext import data, datasets

# import spacy
import numpy as np

import random
import math
import time

import copy

from argparse import ArgumentParser

from collections import Counter




global_test_flag = False

# global_test_flag = True


parser = ArgumentParser(description='G2P Transformer with pretrain MaskLM as Encoder')


parser.add_argument('--debug', action='store_true') 


parser.add_argument('--froze_bert_flag',
                    type=int,
                    default=0)  # if froze_bert_flag is True, this model becomes GBERT w/o finetuning instead of GBERT finetuning.


parser.add_argument('--bert_fine_tuning_lr',
                    type=float,
                    default=3e-5)  # the lr of GBERT encoder

parser.add_argument('--bert_warmup_step_epoch',
                    type=int,
                    default=160)  # The epoch number of the warmup step of GBERT encoder. Following Liu, this is larger than the epoch number of the Transformer decoder


parser.add_argument('--decoder_fine_tuning_lr',
                    type=float,
                    default=1e-3)  #  the lr of Transformer decoder

parser.add_argument('--decoder_warmup_step_epoch',
                    type=int,
                    default=80)  # The epoch number of the warmup step of Transformer decoder

parser.add_argument('--batch_size',
                    type=int,
                    default=256)  # batch size

parser.add_argument('--enc_dec_attention_hid_dim_match_flag',
                    type=int,
                    default=1)  # whether the hid dim of the Transformer decoder and GBERT is matched. this is matched for medium-resource (instead Korean) but not for low-resource.

parser.add_argument('--monolingual_language',
                    type=str,
                    default='dut')  # language


config = parser.parse_args()

# monolingual language
monolingual_language = config.monolingual_language

print('\nconfig')
print(config)

if config.debug:
    global_test_flag = True

    print('\ndebug !!!')


def tokenize_grapheme(enhanced_word):
    """
    tokenizer word with langid (For multilingual input, e.g., <eng>hello)
    """

    index_over = enhanced_word.index('}')
    lang_id = [enhanced_word[:index_over + 1]]

    splited_word = [enhanced_word[i] for i in range(index_over + 1, len(enhanced_word))]

    lang_id.extend(splited_word)
    return lang_id


def tokenize_spelling(word):
    """
    tokenizer word without langid (for the input of GBERT, e.g., hello)
    """

    return [xx for xx in word]




def tokenize_phoneme(phoneme):
    """
    tokenizer for phoneme
    """
    return phoneme.split()

# input data type (the langid is used for multilingual model)
# word with langid, spelling, langid for this model, word input for GBERT, langid for GBERT

# example for word 'hello' and its spelling (a monolingual case)
# <eng>hello, (spelling), 0, hello, 0 
SRC = Field(sequential=True, tokenize=tokenize_grapheme,
            eos_token='<eos>', init_token='<sos>',
            lower=True, batch_first=True)

TRG = Field(sequential=True, tokenize=tokenize_phoneme,
            eos_token='<eos>', init_token='<sos>',
            lower=True, batch_first=True)

SPELLING_PRETRAIN = Field(sequential=True, tokenize=tokenize_spelling,
                          eos_token='<eos>', init_token='<sos>',
                          lower=True, batch_first=True)


LANGID_PRETRAIN = Field(sequential=False, use_vocab=False)


LANGID = Field(sequential=False, use_vocab=False)

fields_evaluate = [('merged_spelling', SRC), ('ipa', TRG), ('langid', LANGID), ('spelling_pretrain', SPELLING_PRETRAIN),
                   ('langid_pretrain', LANGID_PRETRAIN)]

langid_with_brace = '{' + monolingual_language + '}'
whole_language_code_dict_reverse_np = {0: langid_with_brace}

print('\nwhole_language_code_dict_reverse_np')
print(whole_language_code_dict_reverse_np)


train_path = monolingual_language + '_train_grapheme_bert_input_five_column.csv'
validation_path = monolingual_language + '_dev_grapheme_bert_input_five_column.csv'
test_path = monolingual_language + '_test_grapheme_bert_input_five_column.csv'


dir_root_path = '.'

data_path = dir_root_path + '/monolingual_medium_resource/monolingual_g2p_grapheme_bert_input_data'

train_data, valid_data, test_data = data.TabularDataset.splits(
    path=data_path, format='csv',
    train=train_path,
    validation=validation_path,
    test=test_path,
    skip_header=True, fields=fields_evaluate)

print(train_data)
print(type(train_data))

# print to check the right numbers of examples
print(f"Number of training examples: {len(train_data.examples)}")
print(f"Number of validation examples: {len(valid_data.examples)}")
print(f"Number of testing examples: {len(test_data.examples)}")


print('tokenized merged_spelling after split')
print('\n tokenized data')
print(vars(train_data.examples[random.randint(1, 10)]))
print('\n')
print(vars(valid_data.examples[random.randint(1, 10)]))
print('\n')
print(vars(test_data.examples[random.randint(1, 10)]))
# build vocab
SRC.build_vocab(train_data, min_freq=5)
TRG.build_vocab(train_data, min_freq=5)

# build vocab and then replace it with the vocab of pretrained GBERT
SPELLING_PRETRAIN.build_vocab(train_data, min_freq=5)

print('\nOriginal SRC vocab')

src_list = SRC.vocab.stoi

i = 0
for x in src_list:
    print(x, end='\t')
    i += 1
    if i >= 5:
        break




src_vocab_extended = SRC.vocab.itos
src_vocab_dict_extended = SRC.vocab.stoi

# the reversed vocab
src_vocab_dict_extended_reversed = {}
for _, key in enumerate(src_vocab_dict_extended):
    src_vocab_dict_extended_reversed[src_vocab_dict_extended[key]] = key

trg_vocab_dict_extended = TRG.vocab.stoi
trg_vocab_dict_extended_reversed = {}
for _, key in enumerate(trg_vocab_dict_extended):
    trg_vocab_dict_extended_reversed[trg_vocab_dict_extended[key]] = key

print(f"Unique tokens in source (grapheme) vocabulary: {len(SRC.vocab)}")
print(f"Unique tokens in target (phoneme) vocabulary: {len(TRG.vocab)}")

# tell TorchText to put the tensors on the GPU or not.

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device_pretrain = device

# load the vocab of the pretrained GBERT model
pretrain_vocab_path = dir_root_path + '/monolingual_medium_resource' +  '/pretrain_model_vocab/' + monolingual_language + '_wikipron_pretrained_random_mask_lm_vocab.pth'

pretrain_vocab = torch.load(pretrain_vocab_path)

spelling_letter_list = SPELLING_PRETRAIN.vocab.itos
spelling_letter_dict = SPELLING_PRETRAIN.vocab.stoi


SPELLING_PRETRAIN.vocab.itos = pretrain_vocab.itos
SPELLING_PRETRAIN.vocab.stoi = pretrain_vocab.stoi
spelling_letter_list = SPELLING_PRETRAIN.vocab.itos
spelling_letter_dict = SPELLING_PRETRAIN.vocab.stoi

src_pretrain_vocab_dict_extended_reversed = {}
for _, key in enumerate(spelling_letter_dict):
    src_pretrain_vocab_dict_extended_reversed[spelling_letter_dict[key]] = key



BATCH_SIZE = min(config.batch_size, 256)  # Accumulated training for limited GPU resources


EQUAL_BATCH_SIZE = config.batch_size

ACCUMULATE_NUMBER = EQUAL_BATCH_SIZE // BATCH_SIZE

RELU_FLAG = True  # the Transformer decoder will use a gelu activate function

LABEL_SMOOTHED_LOSS_FLAG = True

SMOOTHING = 0.1 # We used the label smoothing following the official baseline, the official baseline is implemented by fairseq.


train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    sort=False,
    device=device)



# GBERT
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

        # self.lang_embedding = nn.Embedding(langid_dim, hid_dim)

        self.src_pad_idx = src_pad_idx

        self.layers = nn.ModuleList([EncoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

        # Masked LM fc (this is discarded when finetuning)
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
  
        # position embedding
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)


        src = self.dropout(
            ((self.tok_embedding(src) ) * self.scale) + self.pos_embedding(pos))

        for layer in self.layers:

            src = layer(src, src_mask)

        src_word_prediction = self.grapheme_prediction_layer(src)

        argmax_output = src_word_prediction.argmax(-1)

        # only the contextual output (src) of the GBERT encoder will be used
        return src, src_word_prediction, argmax_output


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

        # src = [batch size, src len, hid dim]

        _src, _ = self.self_attention(src, src, src, src_mask)


        #dropout, residual connection and layer norm
        src = self.layer_norm(src + self.dropout(_src))

        #src = [batch size, src len, hid dim]

        #positionwise feedforward
        _src = self.positionwise_feedforward(src)

        #dropout, residual and layer norm
        src = self.layer_norm(src + self.dropout(_src))

        #src = [batch size, src len, hid dim]

        return src

# the multi-head attention module of Transformer
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

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy = [batch size, n heads, query len, key len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)

        x = torch.matmul(self.dropout(attention), V)


        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim)

        # x = [batch size, query len, hid dim]

        x = self.fc_o(x)

        # x = [batch size, query len, hid dim]

        return x, attention

# this Enc-Dec attention module is used for the low-resource experiments, where the hid_dim of the query (Transformer decoder) is smaller than these of the key and value (GBERT).
class BertMultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim * 2
        self.n_heads = n_heads
        self.head_dim = hid_dim * 2 // n_heads

        # the hid dim of query is 128 while the hid dim of key/value in GBERT is 256
        self.fc_q = nn.Linear(hid_dim, hid_dim * 2)
        self.fc_k = nn.Linear(hid_dim * 2, hid_dim * 2)
        self.fc_v = nn.Linear(hid_dim * 2, hid_dim * 2)

        self.fc_o = nn.Linear(hid_dim * 2, hid_dim)

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

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy = [batch size, n heads, query len, key len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)

        x = torch.matmul(self.dropout(attention), V)

   
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


# the activate function of GBERT
class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        # x = [batch size, seq len, hid dim]


        x = self.dropout(gelu(self.fc_1(x)))
        # x = [batch size, seq len, pf dim]

        x = self.fc_2(x)

        # x = [batch size, seq len, hid dim]

        return x

# the activate function of Transformer decoder
class PositionwiseFeedforwardLayerNew(nn.Module):
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

# Transformer decoder
class Decoder(nn.Module):
    def __init__(self,
                 output_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 max_length = 80):
        super().__init__()

        self.device = device

        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([DecoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask):

        #trg = [batch size, trg len]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, trg len]
        #src_mask = [batch size, src len]

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        #pos = [batch size, trg len]

        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))

        #trg = [batch size, trg len, hid dim]

        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]

        output = self.fc_out(trg)

        #output = [batch size, trg len, output dim]


        return output, attention




class DecoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()

        self.layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)

        # if the hid dim of Transformer decoder and GBERT is not matched, we multiplied the hid dim of Transformer decoder to match them.
        if config.enc_dec_attention_hid_dim_match_flag:
            self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        else:
            self.encoder_attention = BertMultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)

        self.positionwise_feedforward = PositionwiseFeedforwardLayerNew(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg = [batch size, trg len, hid dim]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, trg len]
        # src_mask = [batch size, src len]

        # self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)

        # dropout, residual connection and layer norm
        trg = self.layer_norm(trg + self.dropout(_trg))

        # trg = [batch size, trg len, hid dim]

        # encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)

        # dropout, residual connection and layer norm
        trg = self.layer_norm(trg + self.dropout(_trg))

        # trg = [batch size, trg len, hid dim]

        # positionwise feedforward
        _trg = self.positionwise_feedforward(trg)

        # dropout, residual and layer norm
        trg = self.layer_norm(trg + self.dropout(_trg))

        # trg = [batch size, trg len, hid dim]
        # attention = [batch size, n heads, trg len, src len]

        return trg, attention



# define object to hold intermediate decoder states, sequences, and probabilities for beam search
class Beam(object):
    def __init__(self, trg_index_history, logp, seq_len):

        super(Beam, self).__init__()

        self.trg_index_history = trg_index_history
        self.logp = logp
        self.logp_adj = logp # log probabilities after length normalization
        self.len = seq_len


class Seq2Seq(nn.Module):
    def __init__(self,
                 decoder,
                 trg_pad_idx,
                 output_dim,
                 device):
        super().__init__()

        self.decoder = decoder
        self.trg_pad_idx = trg_pad_idx
        self.output_dim = output_dim
        self.device = device

    def make_trg_mask(self, trg):

        # trg = [batch size, trg len-1]

        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)

        # trg_pad_mask = [batch size, 1, trg len, 1]

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()

        # trg_sub_mask = [trg len, trg len]

        trg_mask = trg_pad_mask & trg_sub_mask

        # trg_mask = [batch size, 1, trg len, trg len]

        return trg_mask

    def make_predicted_trg_mask(self, trg):

        # the gold trg length is unseen.

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()

        # trg_sub_mask = [trg len, trg len]


        trg_mask = trg_sub_mask

        # trg_mask = [batch size, 1, trg len, trg len]

        return trg_mask

    def forward(self, src_pretrain_hidden, src_mask, trg, TRG_EOS_IDX, TRG_PAD_IDX, eos_no_show_print_flag=False,
                teacher_forcing_ratio=1, debug_flag=False, beam_search_size=None):

        # src = [batch size, src len]
        # trg = [batch size, trg len-1]


        batch_size = trg.shape[0]

        enc_src = src_pretrain_hidden


        if teacher_forcing_ratio == 1:
            trg_mask = self.make_trg_mask(trg)

            outputs, attention = self.decoder(trg, enc_src, trg_mask, src_mask)

            argmax_outputs = outputs.argmax(2)
        else:

            trg_len = trg.shape[1] + 1
            max_length = 80 # max prediction length
            pad_cat_tensor = torch.zeros(batch_size, (max_length - (trg_len - 1))).long().to(self.device)
            pad_cat_tensor[:, :] = TRG_PAD_IDX
            trg = torch.cat((trg, pad_cat_tensor), dim=1)
            # new trg_len (actually the max length)
            trg_len = trg.shape[1] + 1
            # predicted trg mask
            trg_mask = self.make_predicted_trg_mask(trg)

            outputs = torch.zeros(batch_size, trg_len - 1, self.output_dim).to(self.device)

            # first input is the EOS token of target, no meaningful tokens in the gold target will be seen for the decoder input
            argmax_outputs = torch.zeros(batch_size, trg_len - 1).long().to(self.device)
            
            trg_all_zero = torch.zeros(batch_size, trg_len - 1).long().to(self.device)
            trg_all_zero[:, 0] = trg[:, 0]
            input = trg_all_zero
            for t in range(trg_len - 1):

                output, attention = self.decoder(input, enc_src, trg_mask, src_mask)

                # output = [batch size, trg len - 1, output dim]
                # attention = [batch size, trg len - 1, src len]

                outputs[:, t] = output[:, t]


                top1 = output[:, t].argmax(1)

                argmax_outputs[:, t] = top1

                if t < trg_len - 1 - 1:

                    trg_all_zero[:, t + 1] = top1

                    input = trg_all_zero

                # break if all EOS tokens are seen in a batch
                loop_end_flag = 1
                for i in range(batch_size):

                    if torch.where(argmax_outputs[i] == TRG_EOS_IDX)[0].shape[0] == 0:
                        loop_end_flag = 0
                        break

                if loop_end_flag:
                    
                    break


        return outputs, argmax_outputs, None

    def forward_beam(self,src_pretrain_hidden, src_mask, trg,  TRG_EOS_IDX, TRG_PAD_IDX, beam_search_size, print_flag=False):

        # src = [batch size, src len]

        alpha = 0.65

        # debug_flag = False

        if print_flag == True:
            # debug_flag = False
            print('\nbeam search size')
            print(beam_search_size)

        batch_size = trg.shape[0]

        selected_batch = random.randint(0, batch_size - 1)

        # src_mask = [batch size, 1, 1, src len]
        # trg_mask = [batch size, 1, trg len, trg len]

        enc_src = src_pretrain_hidden

        if True:

            trg_len = trg.shape[1] + 1
            max_length = 80  # # max prediction length

            SOS_TOKEN_INDEX = 2
            EOS_TOKEN_INDEX = 3  # unk, pad, SOS, EOS, ...


            prediction = torch.zeros(batch_size, max_length).long().to(self.device)


            this_batch_prediction_max_length = 0

            for i in range(batch_size): 

                beams = []  

                END_FLAG = False 

                # if print_flag:
                this_log_list = []


                for t in range(max_length):  
                    if t == 0:
                        # [[SOS]]
                        trg_input = torch.zeros(1, 1).long().to(self.device)
                        trg_input[0, 0] = SOS_TOKEN_INDEX

                        this_trg_mask = self.make_predicted_trg_mask(trg_input)

                        output, attention = self.decoder(trg_input, enc_src[i].unsqueeze(0), this_trg_mask,
                                                             src_mask[i].unsqueeze(0))                            

                        dist = F.softmax(output[0, 0], dim=-1)
                        idxes = torch.argsort(dist, dim=-1, descending=True)

                        for x in range(beam_search_size):
                            if len(beams) < beam_search_size:

                                this_trg_index_history = torch.cat((trg_input, idxes[x].unsqueeze(0).unsqueeze(1)),
                                                                   dim=-1)
                                # [1]
                                this_prob = torch.log(dist[idxes[x]])  
                                this_seq_len = 1

                                this_node = Beam(this_trg_index_history, this_prob, this_seq_len)

                                beams.append(this_node) 

                    else:  
                        candidates = []

                        eos_prediction_num = 0 


                        k = 0
                        for beam in beams:
                            k += 1

                            
                            trg_input = beam.trg_index_history

                            if trg_input[0, -1] == EOS_TOKEN_INDEX:

                                eos_prediction_num += 1

                                continue

                            this_trg_mask = self.make_predicted_trg_mask(trg_input)

                            # output = [batch size=1, trg len=(t+1), output dim]

                            output, attention = self.decoder(trg_input, enc_src[i].unsqueeze(0),
                                                             this_trg_mask,
                                                             src_mask[i].unsqueeze(0))                               
                            # [batch size=1, trg len=t, output dim=468]
                            # [output dim]
                            dist = F.softmax(output[0, t], dim=-1)
                            idxes = torch.argsort(dist, dim=-1, descending=True)

                            for x in range(beam_search_size):


                                this_trg_index_history = torch.cat((trg_input, idxes[x].unsqueeze(0).unsqueeze(1)),
                                                                   dim=-1)
                                # [1]
                                this_prob = beam.logp + torch.log(dist[idxes[x]])
                                this_seq_len = beam.len + 1
                                this_node = Beam(this_trg_index_history, this_prob, this_seq_len)

                                candidates.append(this_node) 
                                candidates[-1].logp_adj = candidates[-1].logp / (
                                            ((5 + candidates[-1].len) / 6) ** alpha)

                        candidates.sort(key=lambda x: x.logp_adj, reverse=True)
                        if len(candidates) > 0:
                            for y in range(beam_search_size):
                                if len(beams) < beam_search_size:  
                                    beams.append(candidates.pop(0))
                                elif beams[y].trg_index_history[0, -1] == EOS_TOKEN_INDEX:  
                                    try:
                                        if beams[y].logp_adj < candidates[0].logp_adj:  
                                            beams[y] = candidates.pop(0)
                                    except:
                                        
                                        import pdb;
                                        pdb.set_trace()
                                else:
                                    beams[y] = candidates.pop(0)

                        beams.sort(key=lambda x: x.logp_adj, reverse=True)

                        if eos_prediction_num == beam_search_size:
                            END_FLAG = True

                    if END_FLAG: 
                        break


                this_prediction = beams[0].trg_index_history[0][1:]
                this_prediction_len = this_prediction.shape[0]

                prediction[i, :this_prediction_len] = this_prediction

                if this_prediction_len > this_batch_prediction_max_length:
                    this_batch_prediction_max_length = this_prediction_len
           
            # cutted by the predicted max length
            prediction = prediction[:, :this_batch_prediction_max_length]

            # [batch size, max_prediction_len]

            return prediction




# WER, if the length of the gold and the prediction is not the same or not all tokens are exact match, output 1 else output 0.
def WER_loss(predicted, gold, eos_token, print_flag=False):

    assert len(predicted) == len(gold)
    predicted_numpy = predicted.data.cpu().numpy()
    gold_numpy = gold.data.cpu().numpy()

    # cutted by EOS
    gold_numpy = list(gold_numpy)
    for i in range(len(gold_numpy)):
        gold_numpy[i] = list(gold_numpy[i])
        if eos_token not in gold_numpy[i]:
            continue
        else:
            index_i = gold_numpy[i].index(eos_token)
            gold_numpy[i] = gold_numpy[i][:index_i]
        gold_numpy[i] = np.array(gold_numpy[i])
    gold_numpy = np.array(gold_numpy)
    # used the list to avoid some bugs of numpy
    predicted_numpy = list(predicted_numpy)
    for i in range(len(predicted_numpy)):
        predicted_numpy[i] = list(predicted_numpy[i])
        if eos_token not in predicted_numpy[i]:
            continue
        else:
            index_i = predicted_numpy[i].index(eos_token)
            predicted_numpy[i] = predicted_numpy[i][:index_i]
        predicted_numpy[i] = np.array(predicted_numpy[i])
    predicted_numpy = np.array(predicted_numpy)


    trg_num = len(gold_numpy)

    # count unmatched number
    wer_list = []
    unmatched_num = 0

    # print('\n trg_num')
    # print(trg_num)
    for i in range(trg_num):
        unmatched_flag = 0
        if len(gold_numpy[i]) != len(predicted_numpy[i]):  # different length
            unmatched_num += 1
            unmatched_flag = 1
            wer_list.append(unmatched_flag) 
            continue
        else:
            if not (predicted_numpy[i] == gold_numpy[i]).all():  # the same length but not exact match
                unmatched_num += 1
                unmatched_flag = 1

            wer_list.append(unmatched_flag) # other cases, the default unmatch_flag is 0 (matched)
        
    return unmatched_num / trg_num, wer_list



def levenshtein(a, b):
    """
    Why is dynamic programming always so ugly?
    """
    d = [[0 for i in range(len(b) + 1)] for j in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        d[i][0] = i
    for j in range(1, len(b) + 1):
        d[0][j] = j
    for j in range(1, len(b) + 1):
        for i in range(1, len(a) + 1):
            cost = int(a[i - 1] != b[j - 1])
            d[i][j] = min(d[i][j - 1] + 1,
                          d[i - 1][j] + 1, d[i - 1][j - 1] + cost)
    # print(d[len(a)][len(b)])
    return d[len(a)][len(b)]


def PER_loss(predicted, gold, eos_token):
    assert len(predicted) == len(gold)
    predicted_numpy = predicted.data.cpu().numpy()
    gold_numpy = gold.data.cpu().numpy()

    gold_numpy_old = copy.deepcopy(gold_numpy)

    # cutted by EOS
    gold_numpy = list(gold_numpy)
    for i in range(len(gold_numpy)):
        gold_numpy[i] = list(gold_numpy[i])
        if eos_token not in gold_numpy[i]:
            continue
        else:
            index_i = gold_numpy[i].index(eos_token)
            gold_numpy[i] = gold_numpy[i][:index_i]
        gold_numpy[i] = np.array(gold_numpy[i])
    gold_numpy = np.array(gold_numpy)
    
    # used the list to avoid some bugs of numpy
    predicted_numpy = list(predicted_numpy)
    for i in range(len(predicted_numpy)):
        predicted_numpy[i] = list(predicted_numpy[i])
        if eos_token not in predicted_numpy[i]:
            continue
        else:
            index_i = predicted_numpy[i].index(eos_token)
            predicted_numpy[i] = predicted_numpy[i][:index_i]
        predicted_numpy[i] = np.array(predicted_numpy[i])
    predicted_numpy = np.array(predicted_numpy)


    per = 0
    per_list = []
    no_zero_len_number = 0

    per_numerator = 0
    per_denominator = 0

    per_numerator_list = []
    per_denominator_list = []

    for i in range(len(predicted_numpy)):
        if len(gold_numpy[i]) != 0:
            edit_distance = levenshtein(predicted_numpy[i], gold_numpy[i])
            this_per = edit_distance / len(gold_numpy[i])
            per += this_per
            per_list.append(this_per)
            no_zero_len_number += 1

            per_numerator += edit_distance
            per_denominator += len(gold_numpy[i])

            per_numerator_list.append(edit_distance)
            per_denominator_list.append(len(gold_numpy[i]))
    per /= no_zero_len_number

    # return per denominator list, per numerator list for computing sum(Levenshtein) / sum(length)

    return per, per_list, per_denominator, per_numerator, per_denominator_list, per_numerator_list

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

# Hyperparameters
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
HID_DIM = 256
# ENC_LAYERS = 4
DEC_LAYERS = 3
# ENC_HEADS = 3
DEC_HEADS = 4
ENC_PF_DIM = 1024
DEC_PF_DIM = 1024
# ENC_DROPOUT = 0.3
DEC_DROPOUT = 0.2

# OUTPUT_DIM = 127

# enc = Encoder(INPUT_DIM,
#               HID_DIM,
#               ENC_LAYERS,
#               ENC_HEADS,
#               ENC_PF_DIM,
#               ENC_DROPOUT,
#               device)

dec = Decoder(OUTPUT_DIM,
              HID_DIM,
              DEC_LAYERS,
              DEC_HEADS,
              DEC_PF_DIM,
              DEC_DROPOUT,
              device)
# SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]


model = Seq2Seq( dec, TRG_PAD_IDX, OUTPUT_DIM, device).to(device)

model.apply(initialize_weights)

print('model parameters (GBERT finetuning)')
print(model)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')

input_dim_dict = {'bul': 35, 'dut': 39, 'hbs': 63, 'kor': 56, 'eng': 50}
INPUT_DIM = input_dim_dict[monolingual_language]  # the grapheme vocabulary of GBERT

LANGID_DIM = 1


HID_DIM = 256
ENC_LAYERS = 6  
# DEC_LAYERS = 6
ENC_HEADS = 4
# DEC_HEADS = 4
ENC_PF_DIM = 1024
# DEC_PF_DIM = 1024
ENC_DROPOUT = 0.1
# DEC_DROPOUT = 0.1

# OUTPUT_DIM = 127

SRC_PAD_IDX = 1


TRG_PAD_IDX = 4



pretrain_encoder = MaskedLMEncoder(INPUT_DIM,
                      LANGID_DIM,
                      SRC_PAD_IDX,
                      HID_DIM,
                      ENC_LAYERS,
                      ENC_HEADS,
                      ENC_PF_DIM,
                      ENC_DROPOUT,
                      device)

pretrain_model = pretrain_encoder.to(device_pretrain)

# load pre-trained model
PRETRAIN_MODEL_PATH = dir_root_path + '/monolingual_medium_resource' + '/torch_models/g2p-model_' + monolingual_language + '_without_dev_test_g2p_data_random_mask_valid_mask_acc.pt'

pretrain_model.load_state_dict(torch.load(PRETRAIN_MODEL_PATH))

print('Pretrain model')
print(pretrain_model)

print(f'Pretrain model has {count_parameters(pretrain_model):,}  parameters')

if config.froze_bert_flag:
    print('\npretrain_model.requires_grad=False-----------------------------------------')
    for param in pretrain_model.parameters():
        param.requires_grad = False

criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)


TRG_EOS_IDX = 3

# This is computed for the evalutation of the multilingual case
# lang list/ the numerator/dominator of WER/PER in a batch
def wer_per_united(src_langid, wer_list, per_list,per_numerator_list,  per_denominator_list, src_vocab_dict_extended_reversed):
    lang_code_list = []
    wer_sum_new_list = []
    per_sum_new_list = []

    lang_each_number_list = []

    per_sum_numerator_list = []
    per_sum_denominator_list = []

    for i in range(len(src_langid)):
        this_lang_id = src_langid[i]
        this_wer = wer_list[i]

        this_per = per_list[i]

        this_lang_code = this_lang_id  

        # Count the existing lang code list
        if this_lang_code not in lang_code_list:
            lang_code_list.append(this_lang_code)
            wer_sum_new_list.append(this_wer)
            per_sum_new_list.append(this_per)

            per_sum_numerator_list.append(per_numerator_list[i])
            per_sum_denominator_list.append(per_denominator_list[i])
            lang_each_number_list.append(1)
        else:  
            index_this_lang = lang_code_list.index(this_lang_code)
            average_wer = wer_sum_new_list[index_this_lang]
            average_per = per_sum_new_list[index_this_lang]

            wer_sum_new_list[index_this_lang] += this_wer
            per_sum_new_list[index_this_lang] += this_per
            per_sum_numerator_list[index_this_lang] += per_numerator_list[i]
            per_sum_denominator_list[index_this_lang] += per_denominator_list[i]
            lang_each_number_list[index_this_lang] += 1

    # return the numerator/denominator of WER/PER in a batch
    return lang_code_list, wer_sum_new_list, per_sum_new_list, per_sum_numerator_list, per_sum_denominator_list, lang_each_number_list




# This is computed for the evalutation of the multilingual case
# lang list/ the numerator/dominator of WER/PER for already computed batch, a batch new come will update for the whole message
def wer_per_update(languages_code_division_list,languages_division_list, src_langid, wer_list, per_list,per_numerator_list,  per_denominator_list, src_vocab_dict_extended_reversed):


    lang_code_list, wer_sum_new_list, _, per_sum_numerator_list, per_sum_denominator_list, lang_each_number_list = wer_per_united(src_langid, wer_list, per_list,per_numerator_list,  per_denominator_list, src_vocab_dict_extended_reversed)



    index_choosen = random.randint(0, len(lang_code_list)-1)
    for j in range(len(lang_code_list)): 

        this_lang_code = lang_code_list[j]
        this_sum_wer = wer_sum_new_list[j]
        this_sum_per_numerator = per_sum_numerator_list[j]
        this_sum_per_denominator = per_sum_denominator_list[j]
        this_lang_number =  lang_each_number_list[j]


        this_lang_code_count = monolingual_language
        flag_1 = this_lang_code_count in languages_code_division_list[0]

        # for unseen language, this may happen for testing unseen language in a multilingual training case

        if  not (flag_1):

            exit(0)

        for i in range(len(languages_division_list)):  # update the numerator/dominator of WER/PER for each language, since a new batch come.

            language_message_set = languages_division_list[i]
            lang_code_set  = languages_code_division_list[i]

            if  this_lang_code_count in lang_code_set: 

                index_this_lang = lang_code_set.index(this_lang_code_count)

                languages_division_list[i][index_this_lang][-6] += this_sum_wer
                languages_division_list[i][index_this_lang][-5]  += this_lang_number
                languages_division_list[i][index_this_lang][-4] += this_sum_per_numerator
                languages_division_list[i][index_this_lang][-3] += this_sum_per_denominator

    return  languages_division_list

# print the results of all language in different language family, 
def languages_division_result_print(languages_division_list):


    whole_mean_wer_without_number = 0
    whole_mean_per_without_number = 0

    whole_language_number = 0

    total_record_num = 0 

    # the results of each language
    # [langid, train num, valid num, test num, test num/train num, wer_sum, wer_num, per_edit_sum, per_length_sum, wer, per（To be computed)]
    for i in range(len(languages_division_list)):


        this_language_message_list = languages_division_list[i]

        for j, message_list in enumerate(this_language_message_list): 
            this_sum_wer = message_list[-6]
            this_num = message_list[-5]

            this_sum_per_numerator = message_list[-4]
            this_sum_per_denominator = message_list[-3]

            if this_num != 0:  # seen language, this is done for a multilingual test mode


                # the result for each language
                languages_division_list[i][j][-2] = this_sum_wer / this_num

                languages_division_list[i][j][-1] = this_sum_per_numerator / this_sum_per_denominator 



                print('language: %s, record number: %d, WER %.4f \t PER %.4f \t ' %
                                    (languages_division_list[i][j][0], this_num, languages_division_list[i][j][-2],languages_division_list[i][j][-1]))
                        

                whole_mean_wer_without_number += languages_division_list[i][j][-2]
                whole_mean_per_without_number += languages_division_list[i][j][-1]

                whole_language_number += 1

                total_record_num += this_num 


    mean_wer_whole_language = whole_mean_wer_without_number / whole_language_number
    mean_per_whole_language = whole_mean_per_without_number / whole_language_number
    print('\n\nThe average WER and PER for a multilingual test mode')
    print('Language number: %d \t whole record number: %d \t Average WER %.4f \t AveragePER %.4f' % (whole_language_number, total_record_num, \
            mean_wer_whole_language, mean_per_whole_language))


    return mean_wer_whole_language, mean_per_whole_language

def label_smoothed_nll_loss(prediction, target, epsilon, ignore_index=None, print_flag=False, reduce=True):

    # The label smoothed loss for a sequence labelling question from searching from Google
    lprobs = torch.log(torch.softmax(prediction, dim=-1))

    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)

    nll_loss = -lprobs.gather(dim=-1, index=target)

    nll_loss_original = -lprobs.gather(dim=-1, index=target)

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

# get src mask for input of the GBERT
def make_src_mask(src):
    # src = [batch size, src len]

    src_mask = (src != SRC_PAD_IDX).unsqueeze(1).unsqueeze(2)

    # src_mask = [batch size, 1, 1, src len]

    return src_mask


def train(model, pretrain_model, iterator, optimizer_encoder, optimizer_decoder, criterion, clip, languages_code_division_list,
          languages_division_list):
    model.train()


    if not config.froze_bert_flag:  # if GBERT is not frozen, then GBERT if fine-tuned
        pretrain_model.train()

    epoch_loss = 0

    # The 8 record items of each language： langid, train number, validate number, test number, test/train number, WER,PER, already number
    #*****************************************************************************************************************

    for languages_list in languages_division_list:
        for lang_five_column_list in languages_list:

            lang_five_column_list.extend([0, 0, 0, 0, 0, 0])


    if not config.froze_bert_flag:  # if GBERT is not frozen, then GBERT if fine-tuned
        optimizer_encoder.zero_grad()

    optimizer_decoder.zero_grad()

    selected_batch_index = random.randint(0, 100)

    for i, batch in enumerate(iterator):

        src = batch.merged_spelling

        trg = batch.ipa


        src_spelling_pretrain = batch.spelling_pretrain
        src_langid_pretrain = batch.langid_pretrain


        src_langid_original = batch.langid

        src_langid_index = batch.langid.contiguous().data.cpu().numpy()

        # get the src mask input for GBERT encoder
        src_mask = make_src_mask(src_spelling_pretrain)

        langid_list_reverse_index = []

        for this_item in src_langid_index:
            langid_list_reverse_index.append(whole_language_code_dict_reverse_np[this_item])

        print_flag = False
        if i == 0 or i == selected_batch_index:
            print_flag = True

            batch_size = src.shape[0]
            this_random_index = random.randint(0, batch_size - 1)

            this_src = src[this_random_index].contiguous().data.cpu().numpy()
            this_src_langid = src_langid_original[this_random_index].contiguous().data.cpu().numpy()

            this_src_spelling_pretrain = src_spelling_pretrain[this_random_index].contiguous().data.cpu().numpy()
            this_src_langid_pretrain = src_langid_pretrain[this_random_index].contiguous().data.cpu().numpy()

            print('\nsrc ')
            print(this_src)

            print('src_lang')
            print(this_src_langid)

            print('src(spelling pretrain)')
            print(this_src_spelling_pretrain)

            print('src langid pretrained')
            print(this_src_langid_pretrain)

            this_src_reverse = []
            this_src_langid_reverse = []

            this_src_spelling_pretrain_reverse = []

            pad_index = 1
            this_src_non_pad_position = np.where(this_src != pad_index)[0]
            for item in this_src[this_src_non_pad_position]:
                this_src_reverse.append(src_vocab_dict_extended_reversed[item])
            for item in this_src_spelling_pretrain:
                this_src_spelling_pretrain_reverse.append(src_pretrain_vocab_dict_extended_reversed[item])
            this_src_langid_reverse = whole_language_code_dict_reverse_np[int(this_src_langid)]

            print('\nsrc（reverse)')
            print(this_src_reverse)
            print(''.join(this_src_reverse))
            print('src spelling(reverse)')
            print(this_src_spelling_pretrain_reverse)
            print(''.join(this_src_spelling_pretrain_reverse))
            print('src_lang')
            print(this_src_langid_reverse)

        eos_token_show_flag = False

        # input src to GBERT encoder
        src_spelling_pretrain = src_spelling_pretrain.to(device_pretrain)
        src_langid_pretrain = src_langid_pretrain.to(device_pretrain)

        src_len = src_spelling_pretrain.shape[1]

        src_langid_pretrain = src_langid_pretrain.unsqueeze(1).repeat(1, src_len)

        src_pretrain_hidden, src_pretrain_letter_prediction_output, _ = pretrain_model(src_spelling_pretrain,
                                                                                       src_langid_pretrain)

        src_pretrain_hidden = src_pretrain_hidden.to(device)



        # input the output of GBERT encoder to Transformer decoder
        output, argmax_output, _ = model(src_pretrain_hidden, src_mask, trg[:, :-1],  TRG_EOS_IDX, TRG_PAD_IDX,
                                         eos_token_show_flag)

        src_langid = src[:, 1].contiguous().data.cpu().numpy()

        # The gold unk token is unpredictable for any predicted tokens from the prediction, where the disagree of the unk problem is produced from the tokenizer. 
        # more details can refer to the explaination of the 'explaination_of_evaluation.md'
        unk_index = 0
        gold_result = trg[:, 1:].clone().detach()
        gold_result[torch.where(gold_result == unk_index)] = -1
        gold_result = gold_result.clone().detach()

        # Update the numerator of WER (wer_list)
        # Update the numerator/denominator list of PER (per_denominator_list, per_numerator_list) 
        # Others are not used
        WER, wer_list = WER_loss(argmax_output, gold_result, TRG_EOS_IDX, print_flag)
        PER, per_list, this_per_denominator, this_per_numerator, per_denominator_list, per_numerator_list = PER_loss(
            argmax_output, gold_result, TRG_EOS_IDX)

        # Update the above three item for all languages
        languages_division_list = wer_per_update(languages_code_division_list, languages_division_list,
                                                 langid_list_reverse_index, wer_list, per_list, per_numerator_list,
                                                 per_denominator_list, src_vocab_dict_extended_reversed)

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)

        if not LABEL_SMOOTHED_LOSS_FLAG:
            loss = criterion(output, trg)
        else:
            PAD_INDEX = 1  # label smoothing loss
            loss, _ = label_smoothed_nll_loss(output, trg, SMOOTHING, ignore_index=PAD_INDEX, print_flag=print_flag)

        loss = loss / ACCUMULATE_NUMBER  # accumulated loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        if (i + 1) % ACCUMULATE_NUMBER == 0 or ((i + 1) == len(iterator)):  

            if not config.froze_bert_flag:  # whether to train GBERT 
                optimizer_encoder.step()
                optimizer_encoder.zero_grad()

            optimizer_decoder.step()
            optimizer_decoder.zero_grad()

        epoch_loss += loss.item()

        if global_test_flag and i == 10:  # debug
            break

    # print the average results of all languages
    peter_wer, peter_per = languages_division_result_print(languages_division_list)

    for languages_list in languages_division_list:
        for lang_five_column_list in languages_list:
            # recover the message of the list
            lang_five_column_list = lang_five_column_list[:-6]

    return epoch_loss / len(iterator) * EQUAL_BATCH_SIZE, peter_wer, peter_per


def evaluate(model, pretrain_model,  iterator, criterion, languages_code_division_list,
             languages_division_list):
    model.eval()


    pretrain_model.eval()

    epoch_loss = 0

    # The 8 record items of each language： langid, train number, validate number, test number, test/train number, WER,PER, already number
    #*****************************************************************************************************************

    
    for languages_list in languages_division_list:
        for lang_five_column_list in languages_list:
            lang_five_column_list.extend([0, 0, 0, 0, 0, 0])


    with torch.no_grad():

        selected_batch_index = random.randint(0, 100)

        for i, batch in enumerate(iterator):
            src = batch.merged_spelling
            trg = batch.ipa
            
            src_langid = src[:, 1].contiguous().data.cpu().numpy()

            src_langid_index = batch.langid.contiguous().data.cpu().numpy()

            src_spelling_pretrain = batch.spelling_pretrain
            src_langid_pretrain = batch.langid_pretrain

            # get the src mask input for GBERT encoder
            src_mask = make_src_mask(src_spelling_pretrain)


            src_langid_original = batch.langid

            langid_list_reverse_index = []

            for this_item in src_langid_index:

                langid_list_reverse_index.append(whole_language_code_dict_reverse_np[this_item])

            eos_token_show_flag = False

            if i == 0:
                debug_flag = True
            else:
                debug_flag = False


            src_spelling_pretrain = src_spelling_pretrain.to(device_pretrain)
            src_langid_pretrain = src_langid_pretrain.to(device_pretrain)

            src_len = src_spelling_pretrain.shape[1]

            src_langid_pretrain = src_langid_pretrain.unsqueeze(1).repeat(1, src_len)

            src_pretrain_hidden, src_pretrain_letter_prediction_output, _ = pretrain_model(src_spelling_pretrain,
                                                                                           src_langid_pretrain)


            src_pretrain_hidden = src_pretrain_hidden.to(device)



            output, argmax_output, _ = model(src_pretrain_hidden, src_mask, trg[:, :-1],  TRG_EOS_IDX, TRG_PAD_IDX,
                                             eos_token_show_flag, 0, debug_flag)  # turn off teacher forcing

            # The gold unk token is unpredictable for any predicted tokens from the prediction, where the disagree of the unk problem is produced from the tokenizer. 
            # more details can refer to the explaination of the 'explaination_of_evaluation.md'
            
            unk_index = 0
            gold_result = trg[:, 1:].clone().detach()
            gold_result[torch.where(gold_result == unk_index)] = -1
            gold_result = gold_result.clone().detach()

            # Update the numerator of WER (wer_list)
            # Update the numerator/denominator list of PER (per_denominator_list, per_numerator_list) 
            # Others are not used
            print_flag = False
            WER, wer_list = WER_loss(argmax_output, gold_result, TRG_EOS_IDX, print_flag)
            PER, per_list, this_per_denominator, this_per_numerator, per_denominator_list, per_numerator_list = PER_loss(
                argmax_output, gold_result, TRG_EOS_IDX)

            languages_division_list = wer_per_update(languages_code_division_list, languages_division_list,
                                                     langid_list_reverse_index, wer_list, per_list, per_numerator_list,
                                                     per_denominator_list, src_vocab_dict_extended_reversed)

            # output_dim = output.shape[-1]
            trg_len = trg.shape[1]
            # output = [batch size, max_length, hid_dim]
            # output = output[:,: trg_len -1].contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            # trg = [(trg len - 1) * batch size]
            # output = [(trg len - 1) * batch size, output dim]

            # loss = criterion(output, trg)

            if global_test_flag and i == 10:  # debug
                break


    # print the average results of all languages
    peter_wer, peter_per = languages_division_result_print(languages_division_list)

    for languages_list in languages_division_list:
        for lang_five_column_list in languages_list:
            # recover the message of the list
            lang_five_column_list = lang_five_column_list[:-6]

    return epoch_loss / len(iterator), peter_wer, peter_per


# beam_search
def evaluate_beam(model, pretrain_model,  iterator, criterion, languages_code_division_list,
                  languages_division_list):
    model.eval()

    epoch_loss = 0


    pretrain_model.eval()

    for languages_list in languages_division_list:
        for lang_five_column_list in languages_list:
            
            lang_five_column_list.extend([0, 0, 0, 0, 0, 0])


    beam_search_size = 5

    selected_batch_index = random.randint(0, 10)

    print('\nbeam_seearch size=%d' % beam_search_size)
    print('\nlen(iterator)')
    print(len(iterator))

    with torch.no_grad():

        selected_batch_index = random.randint(0, 100)

        for i, batch in enumerate(iterator):

            # if i % 10 == 0:
            print('\n%d/%d processing' % (i, len(iterator)))

            src = batch.merged_spelling
            trg = batch.ipa

            src_spelling_pretrain = batch.spelling_pretrain
            src_langid_pretrain = batch.langid_pretrain


            src_mask = make_src_mask(src_spelling_pretrain)


            src_langid_original = batch.langid

            src_langid = src[:, 1].contiguous().data.cpu().numpy()

            src_langid_index = batch.langid.contiguous().data.cpu().numpy()


            langid_list_reverse_index = []

            for this_item in src_langid_index:

                langid_list_reverse_index.append(whole_language_code_dict_reverse_np[this_item])

            eos_token_show_flag = False


            print_flag = False
            if i == 0 or i == selected_batch_index:
                print_flag = True

                batch_size = src.shape[0]
                this_random_index = random.randint(0, batch_size - 1)

                this_src = src[this_random_index].contiguous().data.cpu().numpy()
                this_src_langid = src_langid_original[this_random_index].contiguous().data.cpu().numpy()

                this_src_spelling_pretrain = src_spelling_pretrain[this_random_index].contiguous().data.cpu().numpy()
                this_src_langid_pretrain = src_langid_pretrain[this_random_index].contiguous().data.cpu().numpy()

                print('\nsrc ')
                print(this_src)

                print('src_lang')
                print(this_src_langid)

                print('src(spelling pretrain)')
                print(this_src_spelling_pretrain)

                print('src langid pretrained')
                print(this_src_langid_pretrain)

                this_src_reverse = []
                this_src_langid_reverse = []

                this_src_spelling_pretrain_reverse = []

                pad_index = 1
                this_src_non_pad_position = np.where(this_src != pad_index)[0]
                for item in this_src[this_src_non_pad_position]:
                    this_src_reverse.append(src_vocab_dict_extended_reversed[item])
                for item in this_src_spelling_pretrain:
                    this_src_spelling_pretrain_reverse.append(src_pretrain_vocab_dict_extended_reversed[item])
                this_src_langid_reverse = whole_language_code_dict_reverse_np[int(this_src_langid)]

                print('\nsrc（reverse)')
                print(this_src_reverse)
                print(''.join(this_src_reverse))
                print('src spelling(reverse)')
                print(this_src_spelling_pretrain_reverse)
                print(''.join(this_src_spelling_pretrain_reverse))
                print('src_lang')
                print(this_src_langid_reverse)



            src_spelling_pretrain = src_spelling_pretrain.to(device_pretrain)
            src_langid_pretrain = src_langid_pretrain.to(device_pretrain)

            src_len = src_spelling_pretrain.shape[1]

            src_langid_pretrain = src_langid_pretrain.unsqueeze(1).repeat(1, src_len)
        
            src_pretrain_hidden, src_pretrain_letter_prediction_output, _ = pretrain_model(src_spelling_pretrain,
                                                                                           src_langid_pretrain)


            src_pretrain_hidden = src_pretrain_hidden.to(device)


            argmax_output = model.forward_beam(src_pretrain_hidden, src_mask, trg[:, :-1], TRG_EOS_IDX, TRG_PAD_IDX,
                                               beam_search_size, print_flag)  # turn off teacher forcing

            # The gold unk token is unpredictable for any predicted tokens from the prediction, where the disagree of the unk problem is produced from the tokenizer. 
            # more details can refer to the explaination of the 'explaination_of_evaluation.md'
            
            unk_index = 0
            gold_result = trg[:, 1:].clone().detach()
            gold_result[torch.where(gold_result == unk_index)] = -1
            gold_result = gold_result.clone().detach()

            # Update the numerator of WER (wer_list)
            # Update the numerator/denominator list of PER (per_denominator_list, per_numerator_list) 
            # Others are not used
            WER, wer_list = WER_loss(argmax_output, gold_result, TRG_EOS_IDX, print_flag)
            PER, per_list, this_per_denominator, this_per_numerator, per_denominator_list, per_numerator_list = PER_loss(
                argmax_output, gold_result, TRG_EOS_IDX)

            # Update the above three item for all languages
            languages_division_list = wer_per_update(languages_code_division_list, languages_division_list,
                                                     langid_list_reverse_index, wer_list, per_list, per_numerator_list,
                                                     per_denominator_list, src_vocab_dict_extended_reversed)

            # output_dim = output.shape[-1]
            trg_len = trg.shape[1]
            # output = [batch size, max_length, hid_dim]
            # output = output[:,: trg_len -1].contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            # trg = [(trg len - 1) * batch size]
            # output = [(trg len - 1) * batch size, output dim]

            # loss = criterion(output, trg)

            if global_test_flag and i == 10:  # debug
                break


    # print the average results of all languages
    peter_wer, peter_per = languages_division_result_print(languages_division_list)

    for languages_list in languages_division_list:
        for lang_five_column_list in languages_list:
            # recover the message of the list
            lang_five_column_list = lang_five_column_list[:-6]

    return epoch_loss / len(iterator), peter_wer, peter_per


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

# whole language (For bilingual case, you need to add a 'eng')
languages_code_division_list = [
    [monolingual_language]]

# Some subset division of the whole language set
language_list_new = []

for language_name in languages_code_division_list[0]:
    language_list_new.append([language_name])
languages_division_list = [language_list_new]  

N_EPOCHS = 400

if global_test_flag:  
    N_EPOCHS = 1

CLIP = 1

# different lr for GBERT and Transformer decodedr
LR_ENCODER = config.bert_fine_tuning_lr

LR_DECODER = config.decoder_fine_tuning_lr

WARMUP_INIT_LR = 1e-7
WARMUP_UPDATES_ENCODER = config.bert_warmup_step_epoch  
WARMUP_UPDATES_DECODER = config.decoder_warmup_step_epoch  

INVERSE_SQER_LR_FLAG = True


SAVE_PATH_BERT_FINE_TUNING = dir_root_path + '/monolingual_medium_resource' + '/torch_models/g2p-model_' + monolingual_language + '_without_g2p_dev_and_test_word_Transformer_with_BERT_pretrain_Encoder_batch_size256_gelu_hid_dim128_new.pt'

SAVE_PATH_BERT_PRETRAIN = dir_root_path + '/monolingual_medium_resource' + '/torch_models/g2p-model_' + monolingual_language + '_without_g2p_dev_and_test_word_Transformer_with_BERT_pretrain_Encoder_batch_size256_gelu_hid_dim128_new_pretrain.pt'


best_valid_per = float('inf')

print("\nconfig")
print(config)

print('\n BATCH_SIZE : %d,accumulate batch size=%d %d times' % (EQUAL_BATCH_SIZE, BATCH_SIZE, ACCUMULATE_NUMBER))
print('EPOCHS: %d, LR_encoder=%.5f, LR_decoder=%.5f' % (N_EPOCHS, LR_ENCODER, LR_DECODER))
print('Embedding hid dim = %d, pf_dim = %d, n layer: %d, multi head: %d' % (HID_DIM, ENC_PF_DIM, ENC_LAYERS, ENC_HEADS))
print('\n relu_flag: ', RELU_FLAG)

print('\nlabel_smoothing_loss: ', LABEL_SMOOTHED_LOSS_FLAG)
print('smoothing: %.2f' % SMOOTHING)



print('\nGBert encoder + Transformer Decoder ----------------------------------------------------')



if INVERSE_SQER_LR_FLAG:
    print('\nwarmup_lr: Encoder %.8f %d epochs warmup, %d epochs sqrt' % (WARMUP_INIT_LR, WARMUP_UPDATES_ENCODER, N_EPOCHS))
    print('\nwarmup_lr: Decoder %.8f %d epochs warmup, %d epochs sqrt' % (WARMUP_INIT_LR, WARMUP_UPDATES_DECODER, N_EPOCHS))





# Inverse sqrt lr
lrs_encoder = torch.linspace(WARMUP_INIT_LR, LR_ENCODER, WARMUP_UPDATES_ENCODER)

lrs_decoder = torch.linspace(WARMUP_INIT_LR, LR_DECODER, WARMUP_UPDATES_DECODER)
# training loss 
Train_loss_list = []
Valid_loss_list = []
Train_wer_list = []
Valid_wer_list = []
Train_per_list = []
Valid_per_list = []



# training
for epoch in range(N_EPOCHS):

    if INVERSE_SQER_LR_FLAG:
        if epoch < WARMUP_UPDATES_ENCODER:
            lr_encoder = lrs_encoder[epoch]
        else:
            decay_factor = (WARMUP_UPDATES_ENCODER / (epoch + 1)) ** 0.5
            lr_encoder = LR_ENCODER * decay_factor

        if epoch < WARMUP_UPDATES_DECODER:
            lr_decoder = lrs_decoder[epoch]
        else:
            decay_factor = (WARMUP_UPDATES_DECODER / (epoch + 1)) ** 0.5
            lr_decoder = LR_DECODER * decay_factor

    else:
        lr = LR

    optimizer_encoder = optim.Adam(pretrain_model.parameters(), lr=lr_encoder)

    optimizer_decoder = optim.Adam(model.parameters(), lr=lr_decoder)

    start_time = time.time()

    train_loss, train_wer, train_per = train(model, pretrain_model, train_iterator, optimizer_encoder, optimizer_decoder,
                                             criterion, CLIP, languages_code_division_list, languages_division_list)
    valid_loss, valid_wer, valid_per = evaluate(model, pretrain_model, valid_iterator, criterion,
                                                languages_code_division_list, languages_division_list)

    test_loss, test_wer, test_per = evaluate(model, pretrain_model, test_iterator, criterion,
                                             languages_code_division_list, languages_division_list)

    print(f'| Test Loss: {test_loss:.4f} ')
    print(f'\t Test. wer: {test_wer:.4f} |  Test. per: {test_per:7.4f}')


    Train_loss_list.append(train_loss)
    Train_wer_list.append(train_wer)
    Train_per_list.append(train_per)
    Valid_loss_list.append(valid_loss)
    Valid_wer_list.append(valid_wer)
    Valid_per_list.append(valid_per)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_per < best_valid_per:
        best_valid_loss = valid_loss
        best_valid_wer = valid_wer
        best_valid_per = valid_per
        torch.save(model.state_dict(), SAVE_PATH_BERT_FINE_TUNING)

        torch.save(pretrain_model.state_dict(), SAVE_PATH_BERT_PRETRAIN)

    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print('\tlr_encoder: %.8f, lr_decoder: %.8f' % (lr_encoder, lr_decoder))
    print(f'\tTrain Loss: {train_loss:.4f} ')
    print(f'\t Val. Loss: {valid_loss:.4f}')
    print(f'\t Train. wer: {train_wer:.4f} |  Train.per: {train_per:7.4f}')
    print(f'\t Val. wer: {valid_wer:.4f} |  Val. per: {valid_per:7.4f}')
    print(f'\t Test. wer: {test_wer:.4f} |  Test. per: {test_per:7.4f}')

    print('\ntrain loss list')
    print(Train_loss_list)
    print('\nValid loss list')
    print(Valid_loss_list)

    print("\nTrain wer list")
    print(Train_wer_list)
    print('Valid_wer_list')
    print(Valid_wer_list)
    
    print('\nTrain per list')
    print(Train_per_list)
    print('Valid_per_list')
    print(Valid_per_list)

# test set evaluation.
model.load_state_dict(torch.load(SAVE_PATH_BERT_FINE_TUNING))

pretrain_model.load_state_dict(torch.load(SAVE_PATH_BERT_PRETRAIN))

test_loss, test_wer_bert_finetuning, test_per_bert_finetuning = evaluate(model, pretrain_model, test_iterator, criterion,
                                         languages_code_division_list, languages_division_list)

# test_loss, test_wer_bert_finetuning_beam_search, test_per_bert_finetuning_beam_search = 0, 0, 0
test_loss, test_wer_bert_finetuning_beam_search, test_per_bert_finetuning_beam_search = evaluate_beam(model, pretrain_model, test_iterator, criterion,
                                              languages_code_division_list, languages_division_list)


best_valid_per_baseline = min(Valid_per_list)
best_valid_per_epoch_baseline = Valid_per_list.index(best_valid_per_baseline)

print('\nconfig')
print(config)

print('language: %s, bert_froze_flag: %d, lr_encoder(SpanBERT): %.5f, lr_decoder(Random initialized): %.5f' % (
    config.monolingual_language,
    config.froze_bert_flag,
    config.bert_fine_tuning_lr,
    config.decoder_fine_tuning_lr
))
print('\nfinal result')
print('\nbest PER on dev: %.4f, epoch: %d/%d' % (best_valid_per_baseline, best_valid_per_epoch_baseline, N_EPOCHS))


print('\nGBERT finetuning WER: %.4f, PER: %.4f, beam search: WER: %.4f, PER: %.4f' % (
    test_wer_bert_finetuning, test_per_bert_finetuning, test_wer_bert_finetuning_beam_search, test_per_bert_finetuning_beam_search
))

print(f'| Test Loss: {test_loss:.4f} | Test PPL: {math.exp(test_loss):7.4f} |')
print(f'\t Test. wer: {test_wer:.4f} |  Test. per: {test_per:7.4f}')


print('\nTrain loss list')
print(Train_loss_list)
print('\nValid loss list')
print(Valid_loss_list)

print('\nTrain wer list')
print(Train_wer_list)
print('\nValid wer list')
print(Valid_wer_list)

print('\nTrain per list')
print(Train_per_list)
print('\nValid per list')
print(Valid_per_list)


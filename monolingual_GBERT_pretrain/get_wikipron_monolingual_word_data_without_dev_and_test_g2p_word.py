import numpy as np

import pandas as pd

import os

import hangul_jamo  # special processing for Korean

import random



dir_root_path = '.'

wikipron_raw_data_path =  dir_root_path + '/data/wikipron/data/scrape/tsv'

# g2p dev/test path
g2p_root_path = dir_root_path + '/monolingual_medium_resource/monolingual_g2p_data'

selected_languages_list = ['eng', 'bul', 'dut', 'hbs', 'kor']


def korean_grapheme_conversion(grapheme):
    return hangul_jamo.decompose(grapheme)


def korean_df_conversion(df):

    df['spelling'] = df['spelling'].apply(korean_grapheme_conversion)

    return df


def get_merged_spelling_word_part(enhanced_word):

    # get raw word without langid
    index_over = enhanced_word.index('}')


    return enhanced_word[index_over+1:]

def get_monolingual_word_dictionary_record(data_path, selected_languages_list=None):

    data_list = os.listdir(data_path)

    print('\ndata list')
    print(len(data_list))
    print(data_list)

    language_list = []
    for data_name in data_list:
        # langid
        this_langid = data_name[:3]

        language_list.append(this_langid)

    language_list = np.unique(language_list)
    print('\nlanguage type')
    print(len(language_list))
    print(language_list)
    print([x for x in language_list])



    for i, data_name in enumerate(data_list):
        this_langid = data_name[:3]

        this_df_path = data_path + '/' + data_name

        this_df = pd.read_csv(this_df_path, sep='\t', header=None, na_filter=False, encoding='utf-8')

        # change column name
        this_df.columns = ['spelling', 'ipa']

        if this_langid == 'kor':
            korean_special_processing_flag = True

        else:
            korean_special_processing_flag = False

        if i == 0:

            this_df['lang'] = this_langid

            if korean_special_processing_flag:
                this_df = korean_df_conversion(this_df)

            whole_df = this_df[['lang', 'spelling']]

        else:

            this_df['lang'] = this_langid

            if korean_special_processing_flag:
                this_df = korean_df_conversion(this_df)

            whole_df = pd.concat((whole_df, this_df[['lang', 'spelling']]))

    whole_df_len_less_one = whole_df[whole_df['spelling'].str.len() < 2]


    # delete word with single letter
    whole_df = whole_df[whole_df['spelling'].str.len() >= 2]

    # drop duplicates for (broad and filtered)
    whole_df = whole_df.drop_duplicates(subset=['lang', 'spelling'], keep='first', ignore_index=True)
 
    print('whole records (wikipron)')

    print(whole_df.shape[0])
    print(whole_df.head(5))


    record_len_list = whole_df['lang'].value_counts()

    print('\nlanguage / record numbers')

    for index in record_len_list.index:
        print('\n%s, record number: %d' % (index, record_len_list[index]))
    print(record_len_list.values)
    print([x for x in record_len_list.values])

    # selected languages
    selected_record_len_list = record_len_list.loc[selected_languages_list]
    print('\nselected language number')
    print(selected_record_len_list)

    for index in selected_record_len_list.index:
        print('\n%s, record number: %d' % (index, selected_record_len_list[index]))
    print(selected_record_len_list.values)
    print([x for x in selected_record_len_list.values])

    selected_df = whole_df[whole_df['lang'].isin(selected_languages_list)]

    selected_df = selected_df.reset_index(drop=True)

    # record the df for each language
    monolingual_word_dict = {}


    for langid in selected_languages_list:
        this_df = selected_df[selected_df['lang'].isin([langid])]



        #  Data columns: merged_spelling(word with Langid), ipa, langid
        this_g2p_dev_path = g2p_root_path + '/' + langid + '_dev.csv'
        this_g2p_test_path = g2p_root_path + '/' + langid + '_test.csv'
        this_g2p_dev_df = pd.read_csv(this_g2p_dev_path, na_filter=False, encoding='utf-8')
        this_g2p_test_df = pd.read_csv(this_g2p_test_path, na_filter=False, encoding='utf-8')

        # get the word list of the G2P dev and test set
 
        this_g2p_dev_df['spelling'] = this_g2p_dev_df['merged_spelling'].apply(get_merged_spelling_word_part)

        this_g2p_dev_df['lang'] = langid

        this_g2p_dev_word_df = this_g2p_dev_df[['spelling', 'lang']]

        this_g2p_test_df['spelling'] = this_g2p_test_df['merged_spelling'].apply(get_merged_spelling_word_part)

        this_g2p_test_df['lang'] = langid

        this_g2p_test_word_df = this_g2p_test_df[['spelling', 'lang']]
    

        # remove the G2P dev word
        this_df = pd.concat((this_df, this_g2p_dev_word_df, this_g2p_dev_word_df)).drop_duplicates(keep=False)
        this_df = this_df.reset_index(drop=True)

        # remove the G2P test word
        this_df = pd.concat((this_df, this_g2p_test_word_df, this_g2p_test_word_df)).drop_duplicates(keep=False)
        this_df = this_df.reset_index(drop=True)


        this_save_path = dir_root_path + '/monolingual_medium_resource/monolingual_word_dictionary/' + langid + '_word_dictionary.csv'



        this_df.to_csv(this_save_path, sep='\t', index=False, header=None)



def partition_data(df, validation_size):
    """
    validation_size: float: maximum portion of data per language size of the validation set
    returns: a partition of the data into training and validation
    """

    def lang_sample(frame):

        assert frame.shape[0] >= 10

        return frame.sample(frac=1 - validation_size, random_state=0)


    train = df.groupby('lang').apply(lang_sample).reset_index(drop=True, level=0)
    validation = df[~df.index.isin(train.index)]
    print('train unique languages size', train['lang'].unique().size)
    print('validation unique languages size', validation['lang'].unique().size)
    return train, validation


# add langid
def merge_langID_word(dataset, language_code_dict):

    dataset['merged_spelling'] = '{' + dataset['lang'] + '}' + dataset['spelling']

    dataset['langid'] = '{' + dataset['lang'] + '}'



    dataset['langid'] = dataset['langid'].map(language_code_dict)

    assert dataset['langid'].isnull().any() == False

    dataset_new = dataset[['merged_spelling', 'langid']]

    dataset_new.index = list(range(dataset_new.shape[0]))

    return dataset_new




def split_word_data_into_training_and_dev(selected_languages_list):
    # 1. divice the whole word list (without G2P dev and test word list) into train and de set
    train_number_list = []
    validate_number_list = []


    for langid in selected_languages_list:
        this_data_path = dir_root_path + '/monolingual_medium_resource/monolingual_word_dictionary/' + langid + '_word_dictionary.csv'

        this_df = pd.read_csv(this_data_path, sep='\t', names=['lang', 'spelling'], na_filter=False, encoding='utf-8')


        train, validate = partition_data(this_df, 0.1)


        train_number_list.append(train.shape[0])
        validate_number_list.append(validate.shape[0])

        # only single language
        this_langid_with_curly_braces = '{' + langid + '}' 
        this_language_code_dict = {this_langid_with_curly_braces: 0}

        merged_train = merge_langID_word(train, this_language_code_dict)
        merged_validate = merge_langID_word(validate, this_language_code_dict)

        train_path = dir_root_path + '/monolingual_medium_resource/monolingual_word_data/' + langid + '_word_data' + '_train_without_g2p_dev_and_test_word' + '.csv'

        validate_path = dir_root_path + '/monolingual_medium_resource/monolingual_word_data/' + langid + '_word_data' + '_validate_without_g2p_dev_and_test_word' + '.csv'

        print('training_path')
        print(train_path)
        print('validate_path')
        print(validate_path)

        merged_train.to_csv(train_path, index=False)

        merged_validate.to_csv(validate_path, index=False)

    print('\nFinal word list for pretraining GBERT-------------------')
    print('language \t train number \t validation number')

    for i, language in enumerate(selected_languages_list):
        print('lang: %s          \t %d \t %d' % (language, train_number_list[i], validate_number_list[i]))

if __name__ == '__main__':
    # get the word list for pretraining GBERT (remove the word list of the G2P dev and test set)
    get_monolingual_word_dictionary_record(wikipron_raw_data_path, selected_languages_list)

    # split the whole word list to train/dev for pretraining GBERT
    split_word_data_into_training_and_dev(selected_languages_list)


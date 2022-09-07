import re
import numpy as np
import pandas as pd
from typing import List, Dict
import nlpcda
from nlpcda.tools.Basetool import Basetool
from nlpcda.config import similarword_path
import jieba
import synonyms
import logging 

from tqdm import tqdm
from copy import deepcopy


class DataPreprocessor:
    def __init__(
        self, 
        remove_username=True, 
        remove_punctuation=True, 
        to_simplified=True, 
        emoji_to_text=True, 
        split_words=True,
        cut_all=True, 
    ):
        self.remove_username = remove_username
        self.remove_punctuation = remove_punctuation
        self.to_simplified = to_simplified
        self.emoji_to_text = emoji_to_text
        self.split = split_words, cut_all
    
    def __call__(self, x):
        if self.remove_username:
            x = self._remove_username(x)
        if self.remove_punctuation:
            x = self._remove_punctuation(x)
        if self.to_simplified:
            x = self._to_simplified(x)
        if self.emoji_to_text:
            x = self._emoji_to_text(x)
        if self.split[0]:
            x = self._jieba_split(x, self.split[1])
        return x

    @staticmethod
    def _remove_username(x):
        """去除用户名，回复转发等字段。（有的用户名检测不到？）"""
        x = re.sub(r"\B[@|回复|转发微博|\/\/][\w\u4E00-\u9FA5]+[:：?\/\s]", "", x)
        x = re.sub(r"转发微博", "", x)
        return x

    @staticmethod
    def _remove_punctuation(text):
        """去除所有中英文标点。"""
        from string import punctuation as punctuation_eng
        from zhon.hanzi import punctuation as punctuation_chn
        text = re.sub(r'[{}]+'.format(punctuation_eng+punctuation_chn),' ',text)
        return text
    
    @staticmethod
    def _to_simplified(x):
        """转换繁体至简体。"""
        from zhconv import convert
        return convert(x, 'zh-cn')

    @staticmethod
    def _emoji_to_text(x):  # 运行较慢
        """转换emoji表情为中文。"""
        from emojiswitch import demojize
        return demojize(x, delimiters=(" ", " "), lang="zh")

    @staticmethod
    def _jieba_split(x, cut_all=True):
        import jieba
        return ' '.join(jieba.cut(x, cut_all=cut_all))


class WordPositionExchange(Basetool):
    '''随机词语交换。'''
    
    def __init__(self, create_num: int = 5, change_rate: float = 0.05, char_gram: int = 3, seed: int = 1):
        super(WordPositionExchange, self).__init__('', create_num, change_rate, seed)
        self.char_gram = char_gram

    def __replace_one(self, one_sentence: str):
        # 变为字 数组
        # sen_chars = list(one_sentence)
        sen_chars = list(jieba.cut(one_sentence, cut_all=False))
        for i in range(len(sen_chars)):
            if self.random.random() < self.change_rate:
                # 非中文字不动！
                if self.__is_chinese(sen_chars[i]) == False:
                    continue
                # 交换位置
                change_i = self.__cpt_exchange_position(sen_chars, i)
                # 进行交换
                sen_chars[i], sen_chars[change_i] = sen_chars[change_i], sen_chars[i]
        return ''.join(sen_chars)

    def __cpt_exchange_position(self, sen_chars: list, position_i):
        # 计算出交换位置
        i = position_i
        j = position_i
        # 从position_i左边，找到第一个不是中文的位置，or 全是中文则不能超过char_gram范围
        while i > 0 and self.__is_chinese(sen_chars[i]) and abs(i - position_i) < self.char_gram:
            i -= 1
        # 从position_i右边，找到第一个不是中文的位置，or 全是中文则不能超过char_gram范围
        while j < len(sen_chars) - 1 and self.__is_chinese(sen_chars[j]) and abs(j - position_i) < self.char_gram:
            j += 1
        # 不是中文导致的推出，需要撤回位置
        if not self.__is_chinese(sen_chars[i]):
            if i < position_i:
                i += 1
        if not self.__is_chinese(sen_chars[j]):
            if j > position_i:
                j -= 1
        return self.random.randint(i, j)

    def __is_chinese(self, a_chr):
        return u'\u4e00' <= a_chr <= u'\u9fff'

    def replace(self, replace_str: str):
        replace_str = replace_str.replace('\n', '').strip()
        sentences = [replace_str]
        t = 0

        while len(sentences) < self.create_num:
            t += 1
            a_sentence = self.__replace_one(replace_str)

            if a_sentence not in sentences:
                sentences.append(a_sentence)
            if t > self.create_num * self.loop_t / self.change_rate:
                break
        return sentences



class AddSimilarWord(Basetool):
    '''
    句中随意添加近义词，生成【语义重复】类型病句。
    '''

    def __init__(self, base_file=similarword_path, create_num=5, change_rate=0.05, seed=1):
        super(AddSimilarWord, self).__init__(base_file, create_num, change_rate, seed)


    def load_paser_base_file(self):
        combine_dict = {}
        for line in open(self.base_file, "r", encoding='utf-8'):
            seperate_word = line.strip().split(" ")
            # 仅保留近义词和相关词，过滤独立词
            if seperate_word[0].endswith("@") or seperate_word[0][0] in ['A', 'B', 'C', 'D']:
                continue
            num = len(seperate_word)
            for i in range(1, num):
                wi = seperate_word[i]
                # add to user dict
                if len(wi) > 1: self.add_word(wi)
                combine_dict[wi] = seperate_word[1:]
        print('load :%s done' % (self.base_file))
        return combine_dict

    def replace(self, replace_str:str):
        replace_str = replace_str.replace('\n', '').strip()
        seg_list = self.jieba.cut(replace_str, cut_all=False)
        words = list(seg_list)
        sentences = [replace_str]
        t = 0
        while len(sentences) < self.create_num:
            t += 1
            a_sentence = ''
            for word in words:
                a_sentence += self.s1(word)
            if a_sentence not in sentences:
                sentences.append(a_sentence)
            if t > self.create_num * self.loop_t / self.change_rate:
                break
        return sentences

    def s1(self, word:str):
        # 替换所有在combine_dict中的
        if len(word) == 1: return word
        if word in self.base_file_mapobj and self.random.random() < self.change_rate:
            wi = self.random.randint(0, len(self.base_file_mapobj[word]) - 1)
            place = self.base_file_mapobj[word][wi]
            if place == word:
                return word
            return word + place if self.random.random() < 0.5 else place + word
        else:
            return word


class ReplaceWithSynonym(Basetool):
    '''
    随机替换同义词。
    '''

    def __init__(self, base_file=None, create_num=5, change_rate=0.05, seed=1):
        super(ReplaceWithSynonym, self).__init__(base_file, create_num, change_rate, seed)

    def replace(self, replace_str:str, thresh=0.75):
        replace_str = replace_str.replace('\n', '').strip()
        seg_list = self.jieba.cut(replace_str, cut_all=False)
        words = list(seg_list)
        sentences = [replace_str]
        t = 0
        while len(sentences) < self.create_num:
            t += 1
            a_sentence = ''
            for word in words:
                a_sentence += self.retrieve_synonym(word, thresh)
            if a_sentence not in sentences:
                sentences.append(a_sentence)
            if t > self.create_num * self.loop_t / self.change_rate:
                break
        return sentences

    def retrieve_synonym(self, word:str, thresh:float):
        if len(word) == 1:  # 跳过单字
            return word
        if self.random.random() < self.change_rate:
            try:
                syn_results = synonyms.nearby(word, size=1)
                place, score = syn_results[0][1], syn_results[1][1]
                place = place if score > thresh else word
            except:
                place = word
            return place
        return word


class ReplaceWithAntonym(Basetool):
    '''
    随机替换反义词。
    '''

    def __init__(self, base_file='D:/Apps/Anaconda3/envs/general-torch/Lib/site-packages/nlpcda/data/antonym.txt', create_num=5, change_rate=0.05, seed=1):
        super(ReplaceWithAntonym, self).__init__(base_file, create_num, change_rate, seed)

    def load_paser_base_file(self):
        combine_dict = {}
        for line in open(self.base_file, "r", encoding='utf-8'):
            k, v = [its for its in line.strip().split(" ") if its]
            if k not in combine_dict.keys():
                combine_dict[k] = [v]
            else:
                combine_dict[k].append(v)
        print('load :%s done' % (self.base_file))
        return combine_dict

    def replace(self, replace_str:str):
        replace_str = replace_str.replace('\n', '').strip()
        words = list(jieba.cut(replace_str))
        sentences = [replace_str]
        t = 0
        while len(sentences) < self.create_num:
            t += 1
            a_sentence = ''
            for word in words:
                if word in self.base_file_mapobj and self.random.random() < self.change_rate:
                    wi = self.random.randint(0, len(self.base_file_mapobj[word]) - 1)
                    place = self.base_file_mapobj[word][wi]
                else:
                    place = word
                a_sentence += place
            if a_sentence not in sentences:
                sentences.append(a_sentence)
            if t > self.create_num * self.loop_t / self.change_rate:
                break
        return sentences


class SwitchLogicOrder(Basetool):
    def __init__(self, base_file=None, create_num=1, change_rate=1, seed=1, keywords=['并', '并且']):
        super(SwitchLogicOrder, self).__init__(base_file, create_num, change_rate, seed)
        self.keywords = keywords

    def replace(self, replace_str:str, thresh=0.75):
        replace_str = replace_str.replace('\n', '').strip()
        seg_list = self.jieba.cut(replace_str, cut_all=False)
        words = list(seg_list)
        for k in self.keywords:
            if k in words:
                return self.switch_logic(replace_str, k)
        return replace_str

    @staticmethod
    def _split_subsentences(text):
        return [s for s in text.replace('，', '，！').replace('。', '。！').replace('；', '；！').split('！') if s]

    def switch_logic(self, s:str, keyword:str='并'):
        ss = self._split_subsentences(s)
        output_sentences = []
        for sen in ss:
            words = list(self.jieba.cut(sen))
            new_s = []
            for i, w in enumerate(words):
                if i == 0 and w == keyword:
                    output_sentences = output_sentences[:-1] + [words[1:-1] + ['，']] + [[keyword] + output_sentences[-1][:-1] + [sen[-1]]]
                    break
                elif w == keyword:
                    new_s = new_s[:-1] + [words[i+1]] + [keyword, new_s[-1]]
                    break
                else:
                    new_s.append(w)
            output_sentences.append(new_s)
        return ''.join([''.join(new_s) for new_s in output_sentences])


class DataAugmentation:
    def __init__(self, configs_:Dict[str, dict]) -> None:
        logging.info(f'Initialising data augumentatation operations, including: {list(configs_.keys())}')
        self.entity_swap, self.random_del, self.random_swap = None, None, None

        configs = deepcopy(configs_)
        if 'random_entity' in configs.keys():
            self.entity_swap_p = configs['random_entity'].pop('prop')
            self.entity_swap = nlpcda.Similarword(**(configs['random_entity']))
        if 'random_delete_char' in configs.keys():
            self.random_del_p = configs['random_delete_char'].pop('prop')
            self.random_del = nlpcda.RandomDeleteChar(**(configs['random_delete_char']))
        if 'random_swap' in configs.keys():
            self.random_swap_p = configs['random_swap'].pop('prop')
            self.random_swap = ReplaceWithSynonym(**(configs['random_swap']))
        if 'random_swap_order' in configs.keys():
            self.random_swap_order_p = configs['random_swap_order'].pop('prop')
            self.random_swap_order = WordPositionExchange(**(configs['random_swap_order']))
        if 'random_add_similar' in configs.keys():
            self.random_add_similar_p = configs['random_add_similar'].pop('prop')
            self.random_add_similar = AddSimilarWord(**(configs['random_add_similar']))
        if 'random_swap_logic_words' in configs.keys():
            self.random_swap_logic_words_p = configs['random_swap_logic_words'].pop('prop')
            self.random_swap_logic_words = nlpcda.Similarword(**(configs['random_swap_logic_words']))
        if 'random_swap_logic_order' in configs.keys():
            self.random_swap_logic_order_p = configs['random_swap_logic_order'].pop('prop')
            self.random_swap_logic_order = SwitchLogicOrder(**(configs['random_swap_logic_order']))
        if 'random_replace_antonym' in configs.keys():
            self.random_replace_antonym_p = configs['random_replace_antonym'].pop('prop')
            self.random_replace_antonym = ReplaceWithSynonym(**(configs['random_replace_antonym']))


    def aug(self, df_full:pd.DataFrame, permute=True, seed=1024) -> pd.DataFrame:
        df_pos = df_full[df_full.label == 1]
        df_neg = df_full[df_full.label == 0]

        L_pos = len(df_pos)
        if self.entity_swap:
            df_pos_aug = self.aug_single(df_pos, L_pos, self.entity_swap_p, self.entity_swap)
        # if self.random_swap_order:
        #     df_pos_aug = self.aug_single(df_pos_aug, L_pos, self.random_swap_order_p, self.random_swap_order)
        if self.random_del:
            df_pos_aug = self.aug_single(df_pos_aug, L_pos, self.random_del_p, self.random_del)
        if self.random_swap:
            df_pos_aug = self.aug_single(df_pos_aug, L_pos, self.random_swap_p, self.random_swap)

        df_neg_aug = self.split_long_sentence(df_neg, label=0)
        L_neg = len(df_neg_aug)
        if self.random_swap:
            df_neg_aug = self.aug_single(df_neg_aug, L_neg, self.random_swap_p, self.random_swap, new_label=0)
        if self.random_del:
            df_neg_aug = self.aug_single(df_neg_aug, L_neg, self.random_del_p, self.random_del, new_label=1)
        if self.random_swap_order:
            df_neg_aug = self.aug_single(df_neg_aug, L_neg, self.random_swap_order_p, self.random_swap_order, new_label=1)
        if self.random_swap_logic_words:
            df_neg_aug = self.aug_single(df_neg_aug, L_neg, self.random_swap_logic_words_p, self.random_swap_logic_words, new_label=1)
        if self.random_swap_logic_order:
            df_neg_aug = self.aug_single(df_neg_aug, L_neg, self.random_swap_logic_order_p, self.random_swap_logic_order, new_label=1)
        if self.random_add_similar:
            df_neg_aug = self.aug_single(df_neg_aug, L_neg, self.random_add_similar_p, self.random_add_similar, new_label=1)
        if self.random_replace_antonym:
            df_neg_aug = self.aug_single(df_neg_aug, L_neg, self.random_replace_antonym_p, self.random_replace_antonym, new_label=1)
        augmented_df = pd.concat((df_neg_aug, df_pos_aug))
        # ----------------------------------------------
        # with pd.option_context('display.max_rows', None, 'display.max_columns', None, ):
        #     pd.options.display.max_colwidth = 100
        #     display(augmented_df[-200:])
        # ----------------------------------------------
        if permute:
            augmented_df = augmented_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        return augmented_df

    def aug_for_structure_pred(self, df_full:pd.DataFrame, permute=True, seed=1024) -> pd.DataFrame:
        df_pos = df_full[df_full.label == 1]
        df_neg = df_full[df_full.label == 0]

        df_pos_aug = df_pos.copy(deep=True)
        L_pos = len(df_pos_aug)
        # if self.entity_swap:
        #     df_pos_aug = self.aug_single(df_pos_aug, L_pos, self.entity_swap_p, self.entity_swap)
        # if self.random_swap_order:
        #     df_pos_aug = self.aug_single(df_pos_aug, L_pos, self.random_swap_order_p, self.random_swap_order)
        if self.random_del:
            df_pos_aug = self.aug_single(df_pos_aug, L_pos, self.random_del_p, self.random_del, keep_original=False)
        if self.random_swap:
            df_pos_aug = self.aug_single(df_pos_aug, len(df_pos_aug), self.random_swap_p, self.random_swap)
        if self.random_add_similar:
            df_pos_aug = self.aug_single(df_pos_aug, len(df_pos_aug), self.random_add_similar_p, self.random_add_similar)

        df_neg_aug = self.split_long_sentence(df_neg, label=0)
        L_neg = len(df_neg_aug)
        if self.random_swap:
            df_neg_aug = self.aug_single(df_neg_aug, L_neg, self.random_swap_p, self.random_swap, new_label=0)
        if self.random_swap_order:
            df_neg_aug = self.aug_single(df_neg_aug, L_neg, self.random_swap_order_p, self.random_swap_order, new_label=1)
        # if self.random_swap_logic_words:
        #     df_neg_aug = self.aug_single(df_neg_aug, L_neg, self.random_swap_logic_words_p, self.random_swap_logic_words, new_label=1)
        augmented_df = pd.concat((df_neg_aug, df_pos_aug))
        # ----------------------------------------------
        # with pd.option_context('display.max_rows', None, 'display.max_columns', None, ):
        #     pd.options.display.max_colwidth = 100
        #     display(augmented_df[-200:])
        # ----------------------------------------------
        if permute:
            augmented_df = augmented_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        return augmented_df

    def aug_single(self, df:pd.DataFrame, L:int, p:float, tool, new_label=None, keep_original=True) -> pd.DataFrame:
        """input L: original df length. Avoid augmentation on newly constructed data. """
        idx = np.random.choice(range(L), size=int(L*p))
        slice_df = df.iloc[idx]
        transformed_slice_df = self.get_transformed_df(slice_df, tool, new_label)
        return pd.concat((df, transformed_slice_df)) if keep_original else transformed_slice_df

    def text_seq_transform(self, tool, texts:List[str]) -> List[str]:
        out = []
        for text in tqdm(texts):
            transformed_text = tool.replace(text)[-1]
            if transformed_text != text:
                out.append(transformed_text)
        return np.array(out)
    
    def get_transformed_df(self, slice_df:pd.DataFrame, tool, new_label) -> pd.DataFrame:
        if not slice_df.any().any():
            return slice_df
        label, text = slice_df[['label', 'text']].values.T
        transformed_text = self.text_seq_transform(tool, text)
        transformed_slice_df = pd.DataFrame({'text':transformed_text, 'label':label[0]})
        if new_label:
            transformed_slice_df['label'] = new_label
        return transformed_slice_df
    
    def split_long_sentence(self, slice_df:pd.DataFrame, label=0, punctuations=['。', '！', '？']) -> pd.DataFrame:
        def flatten(l):
            return [item for sublist in l for item in sublist]
        
        texts = slice_df.text.values

        outputs = []
        for text in texts:
            sentences = []
            for p in punctuations:
                sentences = [s for s in text.split(p) if s]
                sentences = [s+p if s[-1] not in punctuations else s for s in sentences]
                if len(sentences) > 1:
                    break
            outputs.extend(sentences)
            out_df = pd.DataFrame(data={'label':label, 'text':outputs})
        return out_df
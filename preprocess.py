import re
import numpy as np
import pandas as pd
from typing import List, Dict
import nlpcda
from nlpcda.tools.Basetool import Basetool
import jieba

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


class DataAugmentation:
    def __init__(self, configs:Dict[str, dict]) -> None:
        self.entity_swap, self.random_del, self.random_swap = None, None, None

        if 'random_entity' in configs.keys():
            self.entity_swap_p = configs['random_entity'].pop('prop')
            self.entity_swap = nlpcda.Similarword(**(configs['random_entity']))
        if 'random_delete_char' in configs.keys():
            self.random_del_p = configs['random_delete_char'].pop('prop')
            self.random_del = nlpcda.RandomDeleteChar(**(configs['random_delete_char']))
        if 'random_swap' in configs.keys():
            self.random_swap_p = configs['random_swap'].pop('prop')
            self.random_swap = nlpcda.Similarword(**(configs['random_swap']))
        if 'random_swap_order' in configs.keys():
            self.random_swap_order_p = configs['random_swap_order'].pop('prop')
            self.random_swap_order = WordPositionExchange(**(configs['random_swap_order']))

    def aug(self, df_full:pd.DataFrame, permute=True, seed=1024) -> pd.DataFrame:
        df_pos = df_full[df_full.label == 1]
        df_neg = df_full[df_full.label == 0]

        L = len(df_pos)
        if self.entity_swap:
            df_pos_aug = self.aug_single(df_pos, L, self.entity_swap_p, self.entity_swap)
        # if self.random_swap_order:
        #     df_pos_aug = self.aug_single(df_pos_aug, L, self.random_swap_order_p, self.random_swap_order)
        if self.random_del:
            df_pos_aug = self.aug_single(df_pos_aug, L, self.random_del_p, self.random_del)
        if self.random_swap:
            df_pos_aug = self.aug_single(df_pos_aug, L, self.random_swap_p, self.random_swap)

        df_neg_aug = self.split_long_sentence(df_neg, label=0)
        if self.random_swap_order:
            df_neg_aug = self.aug_single(df_neg_aug, len(df_neg_aug), self.random_swap_order_p, self.random_swap_order, new_label=1)
        augmented_df = pd.concat((df_neg_aug, df_pos_aug))
        # ----------------------------------------------
        # with pd.option_context('display.max_rows', None, 'display.max_columns', None, ):
        #     pd.options.display.max_colwidth = 100
        #     display(augmented_df[-200:])
        # ----------------------------------------------
        if permute:
            augmented_df = augmented_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        return augmented_df

    def aug_single(self, df:pd.DataFrame, L:int, p:float, tool, new_label=None) -> pd.DataFrame:
        """input L: original df length. Avoid augmentation on newly constructed data. """
        idx = np.random.choice(range(L), size=int(L*p))
        slice_df = df.iloc[idx]
        transformed_slice_df = self.get_transformed_df(slice_df, tool, new_label)
        augmented_df = pd.concat((df, transformed_slice_df))
        return augmented_df

    def text_seq_transform(self, tool, texts:List[str]) -> List[str]:
        out = []
        for text in texts:
            transformed_text = tool.replace(text)[-1]
            if transformed_text != text:
                out.append(transformed_text)
        return np.array(out)
    
    def get_transformed_df(self, slice_df:pd.DataFrame, tool, new_label) -> pd.DataFrame:
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
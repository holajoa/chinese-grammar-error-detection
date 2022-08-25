import re
import numpy as np
import pandas as pd
from typing import List, Dict


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


class DataAugmentation:
    def __init__(self, configs:Dict[str, dict]) -> None:
        self.entity_swap, self.random_del = None, None

        if 'random_entity' in configs.keys():
            self.entity_swap_p = configs['random_entity'].pop('prop')
            self.entity_swap = nlpcda.Similarword(**(configs['random_entity']))
        if 'random_delete_char' in configs.keys():
            self.random_del_p = configs['random_delete_char'].pop('prop')
            self.random_del = nlpcda.RandomDeleteChar(**(configs['random_delete_char']))
    
    def aug(self, df_full:pd.DataFrame, permute=True, seed=1117) -> pd.DataFrame:
        df = df_full[df_full.label == 1]
        df_neg = df_full[df_full.label == 0]

        L = len(df)
        if self.entity_swap:
            augmented_df = self.aug_single(df, L, self.entity_swap_p, self.entity_swap)
        if self.random_del:
            augmented_df = self.aug_single(augmented_df, L, self.random_del_p, self.random_del)

        augmented_df = pd.concat((df_neg, augmented_df))
        if permute:
            augmented_df = augmented_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        return augmented_df

    def aug_single(self, df:pd.DataFrame, L:int, p:float, tool) -> pd.DataFrame:
        """input L: original df length. Avoid augmentation on newly constructed data. """
        idx = np.random.choice(range(L), size=int(L*p))
        slice_df = df.iloc[idx]
        transformed_slice_df = self.get_transformed_df(slice_df, tool)
        augmented_df = pd.concat((df, transformed_slice_df))
        return augmented_df

    def text_seq_transform(self, tool, texts:List[str]) -> List[str]:
        out = []
        for text in texts:
            transformed_text = tool.replace(text)[-1]
            out.append(transformed_text)
        return np.array(out)
    
    def get_transformed_df(self, slice_df:pd.DataFrame, tool) -> pd.DataFrame:
        text = slice_df['text'].values
        transformed_text = self.text_seq_transform(tool, text)
        transformed_slice_df = slice_df.drop(columns=['text']).copy(deep=True)
        transformed_slice_df['text'] = transformed_text
        return transformed_slice_df
        
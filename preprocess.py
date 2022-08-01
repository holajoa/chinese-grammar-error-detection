import re


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

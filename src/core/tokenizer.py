import os
import re
import logging
import jieba

class JiebaTokenizerBackend:
    def __init__(self, mode=None):
        self.mode = mode

    def tokenize(self, text):
        if self.mode == "search":
            return list(jieba.cut_for_search(text))
        return list(jieba.cut(text))
    
    def tokenize_batch(self, texts):
        return [self.tokenize(text) for text in texts]
    
class CKIPTokenizerBackend:
    def __init__(self, device=None, user_dict_path=None):
        import torch
        from ckip_transformers.nlp import CkipWordSegmenter

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.ws_driver = CkipWordSegmenter(device=device)
        self.user_terms = self.load_user_dict(user_dict_path) if user_dict_path else set()
        
    def merge_user_terms(self, tokens):
        merged = []
        i = 0
        n = len(tokens)
        
        while i < n:
            found = None
            for j in range(n, i, -1):
                candidate = "".join(tokens[i:j])
                if candidate in self.user_terms:
                    found = candidate
                    i = j
                    break
            if found:
                merged.append(found)
            else:
                merged.append(tokens[i])
                i += 1

        return merged

    def tokenize(self, text):
        tokens = self.ws_driver([text])[0]
        if self.user_terms:
            tokens = self.merge_user_terms(tokens)
        return tokens 
    
    def tokenize_batch(self, texts):
        batch_tokens = self.ws_driver(texts)
        if self.user_terms:
            batch_tokens = [self.merge_user_terms(tokens) for tokens in batch_tokens]
        return batch_tokens
    
    def load_user_dict(self, path):
        user_terms = set()
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                term = parts[0]
                user_terms.add(term)
        return user_terms
    
class Tokenizer:
    def __init__(
        self,
        backend="jieba",
        mode=None,
        stopwords_path=None,
        user_dict_path=None,
        normalize_config=None,
        filter_config=None
    ):
        self.logger = self._setup_logger()
        self.backend_name = backend
        self.mode = mode
        self.stopwords = self._load_stopwords(stopwords_path)
        self.normalize_cfg = normalize_config or {}
        self.filter_cfg = filter_config or {}
        
        if backend == "jieba" and user_dict_path and os.path.exists(user_dict_path):
            self.logger.info(f"Loading user dictionary from {user_dict_path}")
            jieba.load_userdict(user_dict_path)
            
        self.re_punct = re.compile(r"[^\w\s\u4e00-\u9fff]+")
        self.re_emoji = re.compile(r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\u2600-\u26FF\u2700-\u27BF]+")

        if backend == "jieba":
            self.backend = JiebaTokenizerBackend(mode=mode)
            self.logger.info("Initialized Jieba tokenizer backend.")
        elif backend == "ckip":
            self.backend = CKIPTokenizerBackend(user_dict_path=user_dict_path)
            self.logger.info("Initialized CKIP tokenizer backend.")
        else:
            raise ValueError(f"Unsupported tokenizer backend: {backend}")  
        
    def _load_stopwords(self, path):
        if not path or not os.path.exists(path):
            return set()
        with open(path, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f if line.strip())
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('JiebaTokenizer')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _fullwidth_to_halfwidth(self, text):
        """全形 → 半形"""
        res = []
        for char in text:
            code = ord(char)
            if code == 0x3000:       
                code = 0x20
            elif 0xFF01 <= code <= 0xFF5E:  
                code -= 0xFEE0
            res.append(chr(code))
        return "".join(res)
    
    def normalize(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)

        cfg = self.normalize_cfg

        if cfg.get("fullwidth_to_halfwidth", False):
            text = self._fullwidth_to_halfwidth(text)
            
        if cfg.get("remove_emoji", False):
            text = self.re_emoji.sub("", text)

        if cfg.get("strip_whitespace", True):
            text = text.strip()

        if cfg.get("lower", False):
            text = text.lower()

        if cfg.get("remove_punctuation", False):
            text = self.re_punct.sub(" ", text)

        text = re.sub(r"\s+", " ", text)

        return text
    
    def _remove_stopwords(self, tokens):
        if not self.stopwords:
            return tokens
        return [t for t in tokens if t not in self.stopwords]
    
    def _filter_tokens(self, tokens):
        drop_empty = self.filter_cfg.get("drop_empty_token", True)
        min_len = self.filter_cfg.get("min_token_length", 1)

        filtered = []
        for t in tokens:
            if drop_empty and not t.strip():
                continue
            if len(t) < min_len:
                continue
            filtered.append(t)
        return filtered
    
    def tokenize(self, text: str):
        if text is None:
            return []
        # Normalize text
        text = self.normalize(text)
        
        tokens = self.backend.tokenize(text)
        # Remove stopwords
        tokens = self._remove_stopwords(tokens)
        # Filter tokens
        tokens = self._filter_tokens(tokens)
        return tokens
    
    def tokenize_batch(self, texts):
        texts = [self.normalize(text) if text is not None else "" for text in texts]
        batch_tokens = self.backend.tokenize_batch(texts)
        batch_tokens = [self._remove_stopwords(tokens) for tokens in batch_tokens]
        batch_tokens = [self._filter_tokens(tokens) for tokens in batch_tokens]
        return batch_tokens
    
    def __call__(self, text: str):
        return self.tokenize(text)
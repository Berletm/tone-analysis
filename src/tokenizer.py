from razdel import sentenize, tokenize
from pymorphy3 import MorphAnalyzer
from typing import List, Dict
from constants import DATA_PTH, MODELS_PTH, STOP_POS, UNK_TOK, PAD_TOK
import pandas as pd
import os
import re

class Tokenizer:
    def __init__(self):
        self.morph = MorphAnalyzer()
        self.str2tok: Dict[str, int] = {UNK_TOK: 0,
                                      PAD_TOK: 1}
        self.tok2str: Dict[int, str] = {0: UNK_TOK,
                                        1: PAD_TOK}
        self.vocab_size = 0
                
    def fit(self, corpus: pd.DataFrame, max_vocab_size: int) -> None:
        for review in corpus:
            sentences = [sent.text for sent in sentenize(review)]
            for sent in sentences:
                toks = [tok.text for tok in tokenize(sent)]
                
                norm_toks = [
                    res.normal_form 
                    for tok in toks 
                    if (parsed := self.morph.parse(tok)) 
                    and (res := parsed[0]).tag.POS not in STOP_POS
                    and re.search(r'[а-яА-ЯёЁ]', tok)
                ]
                
                for tok in norm_toks:
                    if tok in self.str2tok.keys() or len(tok) < 3: continue
                    if len(self.str2tok) > max_vocab_size: return
                    self.str2tok[tok] = len(self.str2tok)
                    self.tok2str[len(self.tok2str)] = tok
        self.vocab_size = len(self.tok2str)
            
    def tokenize(self, x: str) -> List[int]:
        toks = [tok.text for tok in tokenize(x)]
        norm_toks = [
                    res.normal_form 
                    for tok in toks 
                    if (parsed := self.morph.parse(tok)) 
                    and (res := parsed[0]).tag.POS not in STOP_POS
                    and re.search(r'[а-яА-ЯёЁ]', tok)
                ]
        
        ids = [self.str2tok.get(tok, self.str2tok[UNK_TOK]) for tok in norm_toks]
        
        return ids
    
    def detokenize(self, x: List[int]) -> List[str]:
        return [self.tok2str.get(tok, UNK_TOK) for tok in x]

    def load(self, pth: str) -> None:
        with open(pth, "r") as f:
            for line in f.readlines():
                key, val = line.split()
                self.str2tok[key] = int(val)
                self.tok2str[int(val)] = key
        self.vocab_size = len(self.tok2str)
    
    def save(self, name:str, pth: str) -> None:
        with open(os.path.join(pth, name), "w+") as f:
            for key, val in self.str2tok.items():
                f.write(f"{key} {val}\n")
                        
    
if __name__ == "__main__":
    tokenizer = Tokenizer()
    data = pd.read_parquet(DATA_PTH)
    
    data = data["text"]
    
    tokenizer.fit(data, 30_000)
    tokenizer.save("vocab.dict", MODELS_PTH)
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from typing import Tuple, List
import tqdm 
import os
import pandas as pd

from constants import MODELS_PTH, DATA_PTH
from tokenizer import Tokenizer

tokenizer = Tokenizer()
tokenizer.load(os.path.join(MODELS_PTH, "vocab.dict"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Attention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        self.word_attention_space = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.sent_attention_space = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attention_score_space = nn.Linear(hidden_dim, 1, bias=False)
        
        
    def forward(self, sent_embedding: torch.Tensor, encoded_words: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = encoded_words.size(1)
        sent_embedding_expanded = sent_embedding.unsqueeze(1).repeat(1, seq_len, 1) # repeat sentence vec seq_len times -> sentence mat
        
        energy = torch.tanh(self.sent_attention_space(sent_embedding_expanded) + self.word_attention_space(encoded_words))
        
        attention_scores = self.attention_score_space(energy).squeeze(-1)
        
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoded_words).squeeze(1)
        
        return context_vector, attention_weights
        
        
class ToneRegressor(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, max_score: int):
        super().__init__()
        
        self.dropout = nn.Dropout(0.3)
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        self.sent_embedding = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = Attention(2 * hidden_dim)
        
        self.score_space = nn.Linear(2 * hidden_dim, max_score)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        words_embeddings = self.dropout(self.embeddings(x))
        
        encoded_words, (sent_embedding, _) = self.sent_embedding(words_embeddings)
        
        query = torch.cat((sent_embedding[-2,:,:], sent_embedding[-1,:,:]), dim=1)
        
        context, words_weights = self.attention(query, encoded_words)

        score_logits = self.dropout(self.score_space(context))
        
        score_probs = torch.softmax(score_logits, dim=1)
        
        scores = torch.arange(1, 11, dtype=torch.float32, device=device)
        
        return torch.sum(score_probs * scores, dim=1), words_weights


class ScoreDataset(Dataset):
    def __init__(self, tokenizer: Tokenizer, data: pd.DataFrame):
        super().__init__()
        
        self.data = data
        self.tokenizer = tokenizer
        
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        review, score = self.data.iloc[index]
        
        tokenized_review = self.tokenizer.tokenize(review)
        
        return torch.tensor(tokenized_review, dtype=torch.long), torch.tensor(score, dtype=torch.float32)
    
    def __len__(self) -> int:
        return len(self.data)


def rmse_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2)) 

def visualize_attention_rgb(doc: List[str], weights: torch.Tensor, threshold: float = 0.1) -> None:
    min_w = weights.min().item()
    max_w = weights.max().item()

    if max_w > min_w:
        norm_weights = (weights - min_w) / (max_w - min_w)
    else:
        norm_weights = torch.ones_like(weights)

    for word, w in zip(doc, norm_weights):
        w_val = w.item()

        if w_val < threshold:
            print(f"{word}", end=" ")
            continue

        w_scaled = (w_val - threshold) / (1.0 - threshold)

        r = int(255.0 * (1.0 - w_scaled))
        g = int(255.0 * w_scaled)
        b = 0

        brightness = 0.299 * r + 0.587 * g + 0.114 * b
        text_fg_code = "97" if brightness < 128 else "30"

        print(f"\033[48;2;{r};{g};{b}m\033[{text_fg_code}m{word}\033[0m", end=" ")

    print("\n")

def train(n_epoch:int, model: ToneRegressor, train_loader: DataLoader, val_loader: DataLoader) -> ToneRegressor:    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    model.to(device)
    
    for epoch in range(1, n_epoch + 1):
        train_bar = tqdm.tqdm(train_loader, desc="Training")
        
        total_train_loss = 0.0
        model.train()
        for i, (document, score) in enumerate(train_bar, 1):
            document = document.to(device)
            score = score.to(device)
            
            pred_score, _ = model(document)
            
            loss = rmse_loss(score, pred_score)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            train_bar.set_postfix_str(f"rmse_loss: {total_train_loss / i: .4f}")
        
        val_bar = tqdm.tqdm(val_loader, desc="Validation")
        total_val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for i, (document, score) in enumerate(val_bar, 1):
                document = document.to(device)
                score = score.to(device)
                
                pred_score, weights = model(document)
                
                loss = rmse_loss(score, pred_score)
                
                total_val_loss += loss.item()
                val_bar.set_postfix_str(f"rmse_loss: {total_val_loss / i: .4f}")
                
            print(f"Epoch {epoch} | train_loss: {total_train_loss / len(train_loader): .4f} | val_loss: {total_val_loss / len(val_loader): .4f}")
            
            document, score = next(iter(val_loader))
            document = document.to(device)
            score = score.to(device)
                
            pred_score, weights = model(document)
            doc_ind = torch.randint(0, len(document)-1, (1,)).item()
            doc = document[doc_ind].tolist()
            doc = tokenizer.detokenize(doc)
                        
            visualize_attention_rgb(doc, weights[doc_ind])
    
    return model

def collate_fn(batch):
    texts, scores = zip(*batch)
    
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=1)
    
    scores = torch.tensor(scores, dtype=torch.float)
    
    return texts_padded, scores

if __name__ == "__main__":
    data = pd.read_parquet(DATA_PTH)
    data = data[["text", "rating"]].head(len(data) // 2)
    data["rating"] = data["rating"] * 9 / 5 + 1
    
    model = ToneRegressor(tokenizer.vocab_size, 64, 64, 10)
    
    dataset = ScoreDataset(tokenizer, data)
    
    generator = torch.Generator("cpu")
    generator.manual_seed(42)
    train_ds, val_ds = random_split(dataset, [0.8, 0.2], generator)
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=True, collate_fn=collate_fn)
    
    model = train(10, model, train_loader, val_loader)
    
    torch.save(model, os.path.join(MODELS_PTH, "model.pth"))
    # model = torch.load(os.path.join(MODELS_PTH, "model.pth"), weights_only=False)
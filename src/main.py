import torch
from model import ToneRegressor, Attention, visualize_attention_rgb
import os
from constants import MODELS_PTH, DATA_PTH
from tokenizer import Tokenizer
import pandas as pd

def main() -> None:
    tokenizer = Tokenizer()
    tokenizer.load(os.path.join(MODELS_PTH, "vocab.dict"))
    
    model: ToneRegressor = torch.load(os.path.join(MODELS_PTH, "model.pth"), weights_only=False, map_location="cuda")
    model.eval()

    data = pd.read_parquet(DATA_PTH)
    data = data[["text", "rating"]].head(3)
    data["rating"] = data["rating"] * 9 / 5 + 1
    
    with torch.no_grad():
        for _, row in data.iterrows():
            review, rating = row
            tokenized_review = tokenizer.tokenize(review)
            tensor_review = torch.tensor(tokenized_review, device="cuda").unsqueeze(0)
            pred_rating, weights = model(tensor_review)
            print(f"true rating: {rating} | estimated rating: {pred_rating.item()} | delta: {abs(rating - pred_rating.item())}")
            review = tokenizer.detokenize(tokenized_review)
            visualize_attention_rgb(review, weights[0])
            
if __name__ == "__main__":
    main()
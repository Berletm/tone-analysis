import torch
from model import ToneRegressor, Attention, visualize_attention_rgb
import os
from constants import MODELS_PTH, DATA_PTH
from tokenizer import Tokenizer
import pandas as pd
from razdel import tokenize

def main() -> None:
    tokenizer = Tokenizer()
    tokenizer.load(os.path.join(MODELS_PTH, "vocab.dict"))
    
    model: ToneRegressor = torch.load(os.path.join(MODELS_PTH, "model1.pth"), weights_only=False, map_location="cuda")
    model.eval()

    data = pd.read_parquet(DATA_PTH)
    data = data[["text", "rating"]]
    data["rating"] = data["rating"] * 9 / 5 + 1
    
    review_ids = torch.randint(0, len(data) //2, (3,)).tolist()
    with torch.no_grad():
        for _, row in data.iloc[review_ids].iterrows():
            review, rating = row
            tokenized_review, initial_ids = tokenizer.tokenize(review)
            tensor_review = torch.tensor(tokenized_review, device="cuda").unsqueeze(0)
            pred_rating, weights = model(tensor_review)
            # review = list(tok.text for tok in tokenize(review))
            print(f"true rating: {rating} | estimated rating: {pred_rating.item(): .4f} | delta: {abs(rating - pred_rating.item()): .4f}")
            visualize_attention_rgb(tokenizer.detokenize(tokenized_review), weights[0])
            
if __name__ == "__main__":
    main()
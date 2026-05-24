# Future Experiment Idea: Stock-Identifier Embedding (Approach B)

## Concept
Extend the universal multi-stock model (Approach A) by adding a **learnable stock-identity embedding** to the model input. This lets the model differentiate stocks while still learning shared patterns.

## Architecture Change
```python
class BiLSTMUniversalModel(nn.Module):
    def __init__(self, input_size, num_stocks, embed_dim=8, hidden_size=64, ...):
        self.stock_embedding = nn.Embedding(num_stocks, embed_dim)
        self.lstm = nn.LSTM(input_size + embed_dim, hidden_size, ...)
```

Input shape changes from `[batch, window, F]` to `[batch, window, F + embed_dim]`.

## Data Pipeline Impact
- Must carry `stock_ids` array alongside `X` and `y` through windowing → DataLoader
- Forward pass becomes `model(X_batch, stock_ids_batch)`

## Research Value
- Compare Approach A (stock-agnostic) vs B (stock-aware) metrics
- Visualize/cluster learned stock embeddings (t-SNE) — do banks cluster? do tech stocks?
- Answer: "Does stock identity matter for return prediction, or are patterns truly universal?"

## Pros
- Small dimensionality increase (8-dim embedding vs 20 one-hot)
- Embeddings are learnable — model discovers stock similarities
- Interesting thesis content

## Cons
- Doesn't generalize to unseen stocks (fixed embedding table)
- Model may over-rely on identity instead of learning general patterns

## Status
**Parked** — implement after Approach A results are validated.

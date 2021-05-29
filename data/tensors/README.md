# Sample Raw Data

## Example Usage
```python
from pyCP_APR.datasets import load_dataset

# Load a sample authentication tensor
data = load_dataset(name="TOY")

# Training set
coords_train, nnz_train = data['train_coords'], data['train_count']

# Test set and the labels
coords_test, nnz_test = data['test_coords'], data['test_count']
```

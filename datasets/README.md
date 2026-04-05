# Datasets

This directory contains datasets for the research project:
**Topological Drift Detection for Continuous Monitoring of LLM Embedding Streams**

Data files are NOT committed to git due to size. Follow the download instructions below to recreate locally.

---

## Dataset 1: AG News

### Overview
- **Source**: HuggingFace `ag_news`
- **Size**: 120,000 training / 7,600 test samples
- **Format**: HuggingFace Dataset (Arrow/Parquet)
- **Task**: 4-class topic classification
- **Classes**: World (0), Sports (1), Business (2), Sci/Tech (3) — 30,000 each in train
- **License**: Public benchmark

### Why This Dataset
AG News provides 4 semantically distinct news domains with balanced, large samples. It enables:
- **Abrupt drift**: Switch from one news category to another
- **Gradual drift**: Linear mixing of category proportions over time
- **Multi-step drift**: Sequence through all 4 categories

Pre-computed sentence embeddings are available on DTU Figshare (SBERT models: all-distilroberta-v1, all-MiniLM-L12-v2, all-mpnet-base-v2).

### Download Instructions

**Using HuggingFace (recommended):**
```python
from datasets import load_dataset
dataset = load_dataset("ag_news")
dataset.save_to_disk("datasets/ag_news/data")
```

**Loading once downloaded:**
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/ag_news/data")
train = dataset['train']
test = dataset['test']
```

### Sample Data (10 examples in `datasets/ag_news/samples.json`)

```python
# Label mapping
labels = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
# Example
{"text": "Stocks fell sharply on Tuesday...", "label": 2}  # Business
```

### Usage for Drift Experiments

```python
from datasets import load_from_disk
import numpy as np

dataset = load_from_disk("datasets/ag_news/data")

# Get texts by category
category_texts = {}
for label_id in range(4):
    category_texts[label_id] = [
        item['text'] for item in dataset['train']
        if item['label'] == label_id
    ]

# Create abrupt drift stream: N samples from class A, then N from class B
def make_abrupt_stream(class_a, class_b, n=1000):
    texts_a = category_texts[class_a][:n]
    texts_b = category_texts[class_b][:n]
    return texts_a + texts_b, [0]*n + [1]*n  # texts, drift_labels
```

---

## Dataset 2: 20 Newsgroups

### Overview
- **Source**: HuggingFace `SetFit/20_newsgroups`
- **Size**: 11,314 training / 7,532 test samples
- **Format**: HuggingFace Dataset
- **Task**: 20-class newsgroup topic classification
- **Classes**: 20 fine-grained news topics (sci.crypt, talk.politics.guns, rec.sport.hockey, etc.)

### Why This Dataset
Provides 20 semantically related and distinct domains. Enables drift experiments between:
- Related domains (e.g., sci.crypt → sci.med — subtle drift)
- Unrelated domains (e.g., rec.sport.hockey → talk.religion.misc — strong drift)

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("SetFit/20_newsgroups")
dataset.save_to_disk("datasets/20newsgroups/data")
```

**Loading:**
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/20newsgroups/data")
```

---

## Dataset 3: DBpedia14

### Overview
- **Source**: HuggingFace `dbpedia_14`
- **Size**: 560,000 training / 70,000 test samples
- **Format**: HuggingFace Dataset
- **Task**: 14-class entity classification
- **Classes**: Company, EducationalInstitution, Artist, Athlete, OfficeHolder, MeanOfTransportation, Building, NaturalPlace, Village, Animal, Plant, Album, Film, WrittenWork

### Why This Dataset
Very large scale; 14 semantically diverse domains; good for:
- High-volume streaming drift simulation
- Experiments requiring large reference windows

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("dbpedia_14")
dataset.save_to_disk("datasets/dbpedia14/data")
```

---

## Generating Embeddings

Once datasets are downloaded, generate sentence embeddings for drift experiments:

```python
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
import numpy as np

# Load dataset
dataset = load_from_disk("datasets/ag_news/data")

# Load embedding model (choose one)
model = SentenceTransformer('all-MiniLM-L6-v2')   # Fast, 384-dim
# model = SentenceTransformer('all-mpnet-base-v2')  # High quality, 768-dim

# Generate embeddings for a subset
texts = dataset['train']['text'][:5000]
embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)
labels = dataset['train']['label'][:5000]

# Save
np.save("datasets/ag_news/embeddings_minilm.npy", embeddings)
np.save("datasets/ag_news/labels.npy", labels)
```

### Pre-computed Embeddings on DTU Figshare
SBERT embeddings for AG News are available at:
- DTU Figshare: Pretrained sentence BERT models AG News embeddings
- Models: all-distilroberta-v1, all-MiniLM-L12-v2, all-mpnet-base-v2

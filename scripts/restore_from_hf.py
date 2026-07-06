from huggingface_hub import snapshot_download
import os

os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/raw", exist_ok=True)

print("Restoring embeddings...")
snapshot_download(
    repo_id="shiv3589/thgnn-embeddings",
    repo_type="dataset",
    local_dir="data/processed"
)
print("Embeddings restored")

print("Restoring datasets...")
snapshot_download(
    repo_id="shiv3589/thgnn-datasets",
    repo_type="dataset",
    local_dir="data/raw"
)
print("Datasets restored")
print("All data restored — ready to train")

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("Restoring checkpoints...")
snapshot_download(
    repo_id="shiv3589/thgnn-checkpoints",
    repo_type="dataset",
    local_dir="checkpoints",
    token=os.environ.get("HF_TOKEN")
)
print("Checkpoints restored")

print("Restoring results...")
snapshot_download(
    repo_id="shiv3589/thgnn-results",
    repo_type="dataset",
    local_dir="results",
    token=os.environ.get("HF_TOKEN")
)
print("Results restored")

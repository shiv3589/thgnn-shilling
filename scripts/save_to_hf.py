from huggingface_hub import HfApi

api = HfApi()

# --- Create private embeddings repo ---
try:
    api.create_repo(
        repo_id="shiv3589/thgnn-embeddings",
        private=True,
        repo_type="dataset"
    )
    print("Repo created")
except Exception as e:
    print(f"Repo exists or error: {e}")

# Upload embeddings
api.upload_folder(
    folder_path="data/processed",
    repo_id="shiv3589/thgnn-embeddings",
    repo_type="dataset",
    commit_message="TH-GNN embeddings — ml1m, video_games, arts_crafts, musical_instruments"
)
print("Embeddings uploaded successfully")

# --- Create private datasets repo ---
try:
    api.create_repo(
        repo_id="shiv3589/thgnn-datasets",
        private=True,
        repo_type="dataset"
    )
    print("Repo created")
except Exception as e:
    print(f"Repo exists or error: {e}")

# Upload raw datasets
api.upload_folder(
    folder_path="data/raw",
    repo_id="shiv3589/thgnn-datasets",
    repo_type="dataset",
    commit_message="TH-GNN raw datasets — ml1m, video_games, arts_crafts, musical_instruments"
)
print("Datasets uploaded successfully")

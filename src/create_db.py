import torch
import paths
import joblib
from sentence_transformers import SentenceTransformer
from utils import read_csv_in_directory, read_json_as_dict

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


def train_mini_rag(
    data_dir_path: str = paths.DATA_DIR,
    schema_dir_path: str = paths.INPUT_SCHEMA_DIR,
    db_file_path: str = paths.DB_FILE_PATH,
):
    """
    Trains a MiniRAG model using the data in the given directory.

    Args:
    - data_dir_path (str): The path to the directory containing the data.
    - schema_dir_path (str): The path to the directory containing the schema file.
    - db_file_path (str): The path to the database file.

    Returns:
    - None
    """
    print("Reading data schema...")
    # Read the schema file
    schema = read_json_as_dict(schema_dir_path)

    print("Reading data file...")
    # Read the data file
    data = read_csv_in_directory(data_dir_path)

    print("Creating MiniRAG database...")
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    ids = data[schema["idField"]["name"]].values
    documents = data[schema["target"]["name"]].values
    embeddings = model.encode(documents, show_progress_bar=True)

    db = {id_: (doc, emb) for id_, doc, emb in zip(ids, documents, embeddings)}

    print("Saving MiniRAG database...")
    joblib.dump(db, db_file_path)

    print("MiniRAG model saved successfully.")


if __name__ == "__main__":
    train_mini_rag()

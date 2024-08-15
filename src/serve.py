import uuid
import paths
import uvicorn
import joblib
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Any, Dict
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from pydantic import BaseModel, Field
from typing import List


class DocumentInstance(BaseModel):
    id: int = Field(..., example=1, description="Unique identifier for the document.")
    text: str = Field(
        ..., example="Document 1 text", description="Text content of the document."
    )


class InferenceRequest(BaseModel):
    instances: List[DocumentInstance] = Field(
        ...,
        example=[
            {"id": 1, "text": "Document 1 text"},
            {"id": 2, "text": "Document 2 text"},
        ],
        description="List of document instances to process.",
    )
    top_n: int = Field(
        ...,
        ge=1,
        example=5,
        description="Number of similar documents to retrieve for each input document.",
    )


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


def get_model_resources(
    db_file_path: str = paths.DB_FILE_PATH,
) -> Dict[Any, np.ndarray]:
    """Load the database file containing the embeddings and load the embedding model.

    Args:
        db_file_path (str): The path to the database file.

    Returns:
        Dict[str, Any]: A dictionary containing the embeddings.
    """
    db = joblib.load(db_file_path)
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    return {"db": db, "model": model}


def generate_unique_request_id():
    """Generates unique alphanumeric id"""
    return uuid.uuid4().hex[:10]


def create_app(model_resources: Dict[str, Any]) -> FastAPI:

    app = FastAPI()

    @app.get("/ping")
    async def ping() -> dict:
        """GET endpoint that returns a message indicating the service is running.

        Returns:
            dict: A dictionary with a "message" key and "Pong!" value.
        """
        print("Received ping request. Service is healthy...")
        return {"message": "Pong!"}

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Any, exc: RequestValidationError
    ) -> JSONResponse:
        """
        Handle validation errors for FastAPI requests.

        Args:
            request (Any): The FastAPI request instance.
            exc (RequestValidationError): The RequestValidationError instance.
        Returns:
            JSONResponse: A JSON response with the error message and a 400 status code.
        """
        err_msg = "Validation error with request data."
        # Log the error
        print(f"{err_msg} Error: {str(exc)}")
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": str(exc), "predictions": None},
        )

    @app.post("/infer", tags=["inference"], response_class=JSONResponse)
    async def infer(request: InferenceRequest) -> dict:
        """POST endpoint that takes input data as a JSON object and returns
        top_n similar documents.

        Args:
            request (InferenceRequest): The request body containing the input data.

        Raises:
            HTTPException: If there is an error during inference.

        Returns:
            dict: A dictionary with document as key and list of similar documents as value.
        """
        try:
            request_id = generate_unique_request_id()
            print(f"Responding to inference request. Request id: {request_id}")
            print("Starting predictions...")
            inference_instances = request.instances
            inference_ids = [doc.id for doc in inference_instances]

            assert len(set(inference_ids)) == len(
                inference_ids
            ), "Document ids must be unique."

            inference_documents = [doc.text for doc in inference_instances]
            n = request.top_n
            db = model_resources["db"]
            db_documents = [db[key][0] for key in db.keys()]
            db_embeddings = np.array([db[key][1] for key in db.keys()])
            inference_embeddings = model_resources["model"].encode(
                inference_documents, show_progress_bar=True
            )
            similarity = cosine_similarity(inference_embeddings, db_embeddings)

            # Get the indices of the top n similar embeddings
            top_n_indices = similarity.argsort()[:, -n:]

            predictions_response = {}
            for i, id in enumerate(inference_ids):
                similar_documents = [db_documents[idx] for idx in top_n_indices[i]]
                predictions_response[id] = similar_documents

            print("Returning predictions...")
            return predictions_response
        except Exception as exc:
            err_msg = f"Error occurred during inference. Request id: {request_id}"
            print(f"{err_msg} Error: {str(exc)}")
            raise HTTPException(
                status_code=500, detail=f"{err_msg} Error: {str(exc)}"
            ) from exc

    return app


def create_and_run_app(model_resources: Dict[str, Any]):
    """Create and run Fastapi app for inference service

    Args:
        model (ModelResources, optional): The model resources instance.
            Defaults to load model resources from paths defined in paths.py.
    """
    app = create_app(model_resources)
    print("Starting service. Listening on port 8080.")
    uvicorn.run(app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    create_and_run_app(model_resources=get_model_resources())

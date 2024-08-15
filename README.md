# MiniRag for similarity search using SentenceTransformer('all-MiniLM-L6-v2')

## Project Description

This repository is an implementation of the re-usable Mini-RAG. It is implemented in flexible way so that it can be used with any documents dataset with the use of CSV-formatted data, and a JSON-formatted data schema file. 
The following are the requirements for using your data with this model:

- The data must be in CSV format (it can be zipped).
- The schema file must contain an idField and target columns.
- The train file must contain an ID field. The train data must also contain a target column.

---

Here are the highlights of this implementation: <br/>

- A **Similarity Search** algorithm built using **SentenceTransformer** package.
  Additionally, the implementation contains the following features:

## Project Structure
The following is the directory structure of the project:
- **`examples/`**: This directory contains example files for the papers dataset. Two files are included: `papers_schema.json` and `papers.csv`.
You can place these files in the `inputs/schema` and `inputs/data/` folders, respectively.
  - **`/inputs/`**: This directory contains all the input files for this project, including the `data` and `schema` files.
  - **`jupyter/`** The directory contains a tutorial jupyter notebook with a small example of how the mini-rag works.
  - **`/db/`**: This directory is used to store the database of documents along with the embeddings.
- **`src/`**: This directory holds the source code for the project. It is further divided into various subdirectories:
  - **`serve.py`**: This script is used to serve the model as a REST API using **FastAPI**. It loads the artifacts and creates a FastAPI server to serve the model. It provides 2 endpoints: `/ping` and `/infer`. The `/ping` endpoint is used to check if the server is running. The `/infer` endpoint is used to make predictions.
  - **`create_db.py`**: This script is used to create the database. It loads the data, generates embeddings and saves the artifacts in the path `./model/`.
  - **`utils.py`**: This script contains utility functions used by the other scripts.
- **`.gitignore`**: This file specifies the files and folders that should be ignored by Git.
- **`requirements.txt`**: This file contains the packages used in this project.
- **`LICENSE`**: This file contains the license for the project.
- **`README.md`**: This file (this particular document) contains the documentation for the project, explaining how to set it up and use it.

## Usage
In this section we cover the following:
- How to prepare your data for training and inference
- How to run the model implementation locally
- How to use the inference service
### Preparing your data
- If you plan to run this model implementation on your own dataset, you will need your data in a CSV format. Also, you will need to create a schema file as per the Ready Tensor specifications. The schema is in JSON format, and it's easy to create. You can use the example schema file provided in the `examples` directory as a template.
### To run locally
- Create your virtual environment and install dependencies listed in `requirements.txt`.
- Move the two example files (`papers_schema.json` and `papers.csv`) in the `examples` directory into the `./inputs/schema` and `./inputs/data` folders, respectively (or alternatively, place your custom dataset files in the same locations).
- Run the script `src/create_db.py` to create the database. This will save the db in the path `./db/`.
- Run the script `src/serve.py` to start the inference service, which can be queried using the `/ping` and `/infer` endpoints. The service runs on port 8080.
### Using the Inference Service
#### Getting Predictions
To get predictions for a single sample, use the following command:
```bash
curl -X POST -H "Content-Type: application/json" -d '{
  {
    "top_n": 1,
    "instances": [
        "document 1",
    ]
}' http://localhost:8080/infer
```
The key `instances` contains a list of objects, each of which is a sample for which the prediction is requested. The server will respond with a JSON object containing the predicted similar document for each input record:
```json
{
    "We present NeuroLKH, a novel algorithm that combines deep learning with the strong traditional heuristic Lin-Kernighan-Helsgaun (LKH) for solving Traveling Salesman Problem. Specifically, we train a Sparse Graph Network (SGN) with supervised learning for edge scores and unsupervised learning for node penalties, both of which are critical for improving the performance of LKH. Based on the output of SGN, NeuroLKH creates the edge candidate set and transforms edge distances to guide the searching process of LKH. Extensive experiments firmly demonstrate that, by training one model on a wide range of problem sizes, NeuroLKH significantly outperforms LKH and generalizes well to much larger sizes. Also, we show that NeuroLKH can be applied to other routing problems such as Capacitated Vehicle Routing Problem (CVRP), Pickup and Delivery Problem (PDP), and CVRP with Time Windows (CVRPTW).": [
    "Application of deep learning to NP-hard combinatorial optimization problems is an emerging research trend, and a number of interesting approaches have been published over the last few years. In this work we address robust optimization, which is a more complex variant where a max-min problem is to be solved. We obtain robust solutions by solving the inner minimization problem exactly and apply Reinforcement Learning to learn a heuristic for the outer problem. The minimization term in the inner objective represents an obstacle to existing RL-based approaches, as its value depends on the full solution in a non-linear manner and cannot be evaluated for partial solutions constructed by the agent over the course of each episode. We overcome this obstacle by defining the reward in terms of the one-step advantage over a baseline policy whose role can be played by any fast heuristic for the given problem. The agent is trained to maximize the total advantage, which, as we show, is equivalent to the original objective. We validate our approach by solving min-max versions of standard benchmarks for the Capacitated Vehicle Routing and the Traveling Salesperson Problem, where our agents obtain near-optimal solutions and improve upon the baselines."
    ]
}
```

## Requirements

Dependencies for the main model implementation in `src` are listed in the file `requirements.txt`.
You can install these packages by running the following command from the root of your project directory:

```python
pip install -r requirements.txt
```

## LICENSE

This project is provided under the Apache 2.0 License. Please see the [LICENSE](LICENSE) file for more information.

## Contact Information

Repository created by Ready Tensor, Inc. (https://www.readytensor.ai/)
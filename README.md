
# Chess Piece Classification Project



### Project Structure

```
chessman_classification/
├── data/
│ └── Chessman/
│ ├── Bishop/
│ ├── King/
│ ├── Knight/
│ ├── Pawn/
│ ├── Queen/
│ └── Rook/
├── notebooks/
│ └── best_model.keras
├── src/
│ ├── init.py
│ ├── api.py
│ ├── app.py
│ ├── data_preprocessing.py
│ ├── evaluate.py
│ ├── model.py
│ └── train.py
├── docker/
│ ├── Dockerfile.api
│ ├── Dockerfile.streamlit
│ └── docker-compose.yml
├── requirements.txt
├── README.md
└── report/
└── report.pdf
```




Test (Demo) : https://drive.google.com/file/d/1IqXkpU8tcyTwnbRIPdjbxqe6TpGUcb91/view?usp=sharing




## Objective

Develop a sophisticated image classification model that can identify different chess pieces using the Chessman Image Dataset. This project covers data preprocessing, model training, evaluation, deployment, and Docker containerization. The project includes a REST API and a Streamlit interface for user interaction.

## Prerequisites

Ensure you have the following installed on your machine:

- Docker
- Docker Compose
- Python 3.12
- Git

## Setup Instructions

### Step 1: Clone the Repository

Clone the project repository from GitHub:

```bash
git clone <repository_url>
cd chessman_classification
```

### Step 2: Install Dependencies
Install the required Python packages using the requirements.txt file:
``` bash
pip install -r requirements.txt
```

### Step 3: Download Model weight 

 I used VGG16 as the base model, download the model weight and store in the model directory. link to the model https://huggingface.co/matthias-wright/vgg/blob/main/vgg16_weights.h5


### Step 3: Train the Model

The model can be train in two ways
1.Using the /notebooks/Chessman_Classification.ipynb notebook ; make sure to load the path to the VGG16 weights before running the notebook.
2.Running the Src/ (data_processing.py,model.py,train.py,evaluate.py) script to start the pipelines in that  order which is more structed and production grade.


### Build and run the Docker containers:
```bash

# Navigate to the docker directory 
cd docker


docker-compose up --build
#Starts the containers
```

```bash
docker-compose down
#stops the containers
```


```
Access the Streamlit interface:

Network URL: http://localhost:8501
External URL: http://your-external-ip:8501


Access the FastAPI documentation:

URL: http://localhost:8000/docs


The Predict endoint: http://localhost:8000/docs/predict 


```

### Usage

Upload an image of a chess piece using the Streamlit interface.
The interface will display the classification result along with metrics like latency and throughput.



### Troubleshooting


Common Issues and Solutions
Docker Network Not Found:

Ensure the docker-compose.yml file has the correct network configuration.
```bash
Run docker network create chessman_net to create the network manually.

```

API Not Accessible:

Check if the API container is running using

```docker ps.

```
Verify the API is accessible using 
```
curl -X POST "http://localhost:8000/predict".
````

You can also Navigate to the /src to run the Streamlit and the fastapi directly 

```bash
uvicorn src.api:app --reload
```

```bash
streamlit run src/app.py  
```


Also Ensure, when the Docker containers are up, the streamlit application is sending requests  to the right endpoints:



# GAN Assignment - MNIST Digit Generator API

This project implements a Generative Adversarial Network (GAN) trained on the MNIST dataset to generate handwritten digit images.  
A FastAPI server is provided so that the trained model can be queried via an API endpoint.

## ✅ Features
- GAN implementation in PyTorch
- Trained on MNIST
- Generator model exposed through FastAPI endpoint
- Docker container deployment supported

## 🚀 Run Locally

```bash
uvicorn app.main:app --reload
```

Open the API docs:
http://localhost:8000/docs

Run with Docker

```bash
docker build -t mnist-gan-api .
docker run --rm -p 8000:8000 mnist-gan-api
```

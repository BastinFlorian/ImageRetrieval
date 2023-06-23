from src.model import prepare_model
from src.predict import predict

if __name__ == '__main__':

    url = "https://c4.staticflickr.com/9/8555/15625756039_a60b0bd0a5_o.jpg"
    model, processor, dataset, dataset_with_embeddings = prepare_model()
    id_results, retrieved_examples, query_image = predict(model, processor, dataset, dataset_with_embeddings, url)
    id_results, retrieved_examples, query_image = predict(model, processor, dataset, dataset_with_embeddings, url)
    id_results, retrieved_examples, query_image = predict(model, processor, dataset, dataset_with_embeddings, url)

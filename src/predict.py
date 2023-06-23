from src.utils import score_and_retrieved_examples, retrieved_id, download_image, load_data
from src.const import QUERY_IMAGE_DIR_NAME


def url_to_image(url, dir_name_query_image):
    download_image(url, dir_name_query_image)
    return True


def predict(model, processor, dataset, dataset_with_embeddings, url):
    # Load query_image from query_url
    url_to_image(url, QUERY_IMAGE_DIR_NAME)

    # Create image dataset
    query_image = load_data(QUERY_IMAGE_DIR_NAME)

    # Compute similarities
    scores, retrieved_examples = score_and_retrieved_examples(
        model,
        processor,
        dataset,
        dataset_with_embeddings,
        query_image["image"])

    id_results = retrieved_id(retrieved_examples)

    return id_results, retrieved_examples, query_image

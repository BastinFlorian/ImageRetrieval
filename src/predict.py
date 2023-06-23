from src.utils import score_and_retrieved_examples, retrieved_id, download_image, load_data_from_file
from src.const import QUERY_IMAGE_DIR_NAME
from PIL import Image

def url_to_image(url, dir_name_query_image):
    filename = download_image(url, dir_name_query_image)
    return filename


def predict(model, processor, dataset, dataset_with_embeddings, url):
    # Load query_image from query_url
    filename = url_to_image(url, QUERY_IMAGE_DIR_NAME)

    # Create image dataset
    #query_image = load_data_from_file(filename)
    query_image = Image.open(filename)

    # Compute similarities
    scores, retrieved_examples = score_and_retrieved_examples(
        model,
        processor,
        dataset,
        dataset_with_embeddings,
        query_image
    )

    id_results = retrieved_id(retrieved_examples)

    return id_results, retrieved_examples, query_image

from PIL import Image
from datasets import load_dataset
from transformers import ViTImageProcessor, ViTModel

import requests
import logging
import csv
from pathlib import Path
import pandas as pd
from src.const import HUGGING_FACE_MODEL_NAME
import os


def download_image(url: str, dir_name: str):
    """
    Download image from url and save it to filename
    Args:
        url (type): description
    """
    filename = url.split("/")[-1]
    file = Path(dir_name).joinpath(filename)
    file.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True)
    response.raise_for_status()

    with file.open("wb") as handle:
        for block in response.iter_content(1024):
            if not block:
                break

            handle.write(block)

    return file


def execute_download_image(dir_name: str, csv_name: str):
    """summary

    Args:
        CSV_NAME (type): description

    Returns:
        type: description
    """
    with open(csv_name, "r") as handle:
        reader = csv.reader(handle)
        urls = [r[0] for i, r in enumerate(reader) if i > 0]

    for url in urls:
        try:
            _ = download_image(url, dir_name)
        except requests.exceptions.HTTPError as err:
            logging.warning(f'{url} cannot be read: {err}')
        except Exception as err:
            logging.warning(f'Other error for {url}: {err}')


def load_csv(csvfile_name):
    return pd.read_csv(csvfile_name, sep=",")


def create_metadata(df, data_dir):
    '''
    Metadata for ImageFolder dataset builder
    '''
    url_names = [i.split('/')[-1] for i in df['url']]
    df.drop('url', axis=1, inplace=True)
    df.insert(0, 'file_name', url_names)
    df.to_csv(os.path.join(data_dir, 'metadata.csv'), sep=',', index=False)


def load_data_from_dir(data_dir: str):
    dataset = load_dataset('imagefolder', data_dir=data_dir, split='train', drop_metadata=False)
    return dataset


def load_data_from_file(filename: str):
    dataset = load_dataset('imagefolder', filename, split='train', drop_metadata=False)
    return dataset


def load_model(hugging_face_model_name: str):
    processor = ViTImageProcessor.from_pretrained(HUGGING_FACE_MODEL_NAME)
    model = ViTModel.from_pretrained(HUGGING_FACE_MODEL_NAME)
    return model, processor


def extract_embedding(image, model, processor):
    processed_image = processor(image, return_tensors="pt")
    embedded_image = model(**processed_image).last_hidden_state[:, 0].detach().numpy()
    return embedded_image.squeeze()


def set_index(dataset_with_embeddings):
    dataset_with_embeddings.add_faiss_index(column='embeddings')
    return dataset_with_embeddings


def get_closest_neighbors(query_image, model, processor, dataset_with_embedding, top_k=3):
    embedded_query_image = extract_embedding(query_image, model, processor)
    scores, retrieved_examples = dataset_with_embedding.get_nearest_examples(
        'embeddings',
        embedded_query_image,
        k=top_k
    )

    return scores, retrieved_examples


def score_and_retrieved_examples(model, processor, dataset, dataset_with_embeddings, query_image):
    scores, retrieved_examples = get_closest_neighbors(query_image, model, processor, dataset_with_embeddings, top_k=3)
    return scores, retrieved_examples


def display_retrieved_images(retrieved_examples):
    '''display only results and not the image from the query image'''
    imgs = retrieved_examples["image"]
    w, h = imgs[0].size
    rows = 1
    cols = len(imgs)
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def display_results(imgs, rows, cols):
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def retrieved_id(retrieved_examples):
    return retrieved_examples['id']

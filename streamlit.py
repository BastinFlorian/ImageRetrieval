import streamlit as st
from src.predict import predict
from src import utils
from src.const import IMAGE_DIR, CSV_NAME, HUGGING_FACE_MODEL_NAME


@st.cache_data
def prepare_model():

    utils.execute_download_image(IMAGE_DIR, CSV_NAME)

    df = utils.load_csv(CSV_NAME)
    utils.create_metadata(df, IMAGE_DIR)

    dataset = utils.load_data(IMAGE_DIR)
    model, processor = utils.load_model(HUGGING_FACE_MODEL_NAME)

    dataset_with_embeddings = dataset.map(
      lambda row: {'embeddings': utils.extract_embedding(row["image"], model, processor)}
    )

    dataset_with_embeddings = utils.set_index(dataset_with_embeddings)

    return model, processor, dataset, dataset_with_embeddings


def get_similar_images(model, processor, dataset, dataset_with_embeddings, url):
    id_results, retrieved_examples, input_image = predict(model, processor, dataset, dataset_with_embeddings, url)
    return id_results, retrieved_examples, input_image


st.set_page_config(layout="wide")
st.title("Image retrieval")

with st.spinner('Model creation'):
    model, processor, dataset, dataset_with_embeddings = prepare_model()

with st.form('Input'):
    url = st.text_input("Write an URL", "https://c8.staticflickr.com/9/8602/16049352178_7d3413f8dc_o.jpg")
    submitted = st.form_submit_button('Find similar images')

    if submitted:
        id_results, retrieved_examples, input_image = get_similar_images(
            model,
            processor,
            dataset,
            dataset_with_embeddings,
            url
        )

        st.write(id_results)
        st.image(input_image['image'], caption='The input image', width=600)

        st.image(
            retrieved_examples['image'],
            caption=[f'Top {i} similar' for i in range(len(retrieved_examples['image']))],
            use_column_width="False",
            width=300
        )

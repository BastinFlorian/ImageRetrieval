import streamlit as st
from src.predict import predict
from src.model import prepare_model
from src import utils
from src.const import IMAGE_DIR, CSV_NAME, HUGGING_FACE_MODEL_NAME

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

        st.image(input_image, caption='The input image', width=600)
        st.image(
            retrieved_examples['image'],
            caption=[f'Top {i+1}' for i in range(len(retrieved_examples['image']))],
            use_column_width="False",
            width=300
        )

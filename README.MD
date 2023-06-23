# Image Retrieval

An image retrieval model that gives the IDs of the most similar images given an image
## How to use
----------


- Install pip and Docker
- Run `docker docker build -t image-retrieval .`
- Run `docker run -d --name image-retrieval-container -p 80:80 image-retrieval`
- Open on a web browser: `http://127.0.0.1:8000/docs`
- Alternatively for the Streamlit app, open on a web browser: `http://localhost:8501/`
- Try with the url image of your choice

## How it works
----------

blabla
- FAISS
- HUGGING FACE MODEL NAME

## Limitations and future update
----------


- Performance pas bonnes si beaucoup d'images
  - Dur de faire toutes les similarité malgré utilisation d'un vector database comme FAISS
  - Le stockage en mémoire n'est pas bon car on conserve tous les objets (model, dataset, ..)
  - Long au chargement au départ
    - Axe d'amélioration: reprendre le processus parallélisé de download images


## Licence
----------

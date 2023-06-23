# Image Retrieval

An image retrieval model that gives the IDs of the most similar images given an image
## How to use


- Install pip and Docker
- Run `docker-compose build`
- Run `docker-compose up`
- Fast API app, open on a web browser: `http://localhost:8000/docs`
- Streamlit app, open on a web browser: `http://localhost:8501/`
- Try the apps with the url image of your choice

## How it works


blabla
- FAISS
- HUGGING FACE MODEL NAME

## Limitations and future update


- Performance pas bonnes si beaucoup d'images
  - Dur de faire toutes les similarité malgré utilisation d'un vector database comme FAISS
  - Le stockage en mémoire n'est pas bon car on conserve tous les objets (model, dataset, ..)
  - Long au chargement au départ
    - Axe d'amélioration: reprendre le processus parallélisé de download images


## Licence


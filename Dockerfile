FROM python:3.9

WORKDIR /IR

COPY . .

RUN pip install --no-cache-dir --upgrade -r ./requirements.txt

# # Streamlit
# EXPOSE 8501
# HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
# ENTRYPOINT ["streamlit", "run", "streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Uvicorn
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "80"]
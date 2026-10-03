
FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.streamlit.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY streamlit_app.py .

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

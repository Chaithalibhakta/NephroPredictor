FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

COPY backend/requirements.txt /app/backend/requirements.txt

RUN pip install --no-cache-dir -r /app/backend/requirements.txt

COPY backend /app/backend
COPY models /app/models
COPY segmentation /app/segmentation

# Keep the existing frozen main.py compatible with Linux.
# main.py uses this Windows-style path as a relative path.
RUN mkdir -p "/app/C:\Users\HP\OneDrive\Desktop\NephroPredictorr" && \
    cp -r /app/models "/app/C:\Users\HP\OneDrive\Desktop\NephroPredictorr/" && \
    cp -r /app/segmentation "/app/C:\Users\HP\OneDrive\Desktop\NephroPredictorr/"

EXPOSE 7860

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
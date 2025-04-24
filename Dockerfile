
FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-runtime


WORKDIR /app


COPY . .


RUN apt-get update && apt-get install -y ffmpeg && apt-get clean


RUN pip install --upgrade pip \
 && pip install -r requirements.txt


COPY modelLoader.py /app/modelLoader.py
 RUN python /app/modelLoader.py

# Expose port
EXPOSE 8080

# Command to run your FastAPI app via uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]

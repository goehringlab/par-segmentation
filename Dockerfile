FROM python:3.11.0
COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
COPY ./par_segmentation/ ./par_segmentation
COPY ./scripts/ ./scripts
CMD ["jupyter" "notebook" "--ip" "0.0.0.0"]

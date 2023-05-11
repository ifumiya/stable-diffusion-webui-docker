
# Install sd-scripts ===========================================
FROM python:3.10.6 as sd-scripts

# feat https://github.com/kohya-ss/sd-scripts/blob/main/README-ja.md

RUN git clone https://github.com/kohya-ss/sd-scripts.git
WORKDIR /sd-scripts
RUN pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip install --upgrade -r requirements.txt
RUN pip install -U -I xformers

RUN apt-get update && apt-get install -y libgl1-mesa-dev && apt-get clean

COPY --chmod=111 ./run.sh /sd-scripts/run.sh

ENTRYPOINT [ "/sd-scripts/run.sh" ]

CMD [ "" ]

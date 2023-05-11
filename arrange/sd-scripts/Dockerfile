
# Build xformars package ============================================
FROM python:3.10.6 as xformers

# feat https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Xformers#building-xformers-on-linux-from-anonymous-user

RUN git clone https://github.com/facebookresearch/xformers.git
WORKDIR /xformers

RUN git submodule update --init --recursive
RUN pip install -r requirements.txt
# not have to ?
RUN pip install -e .
RUN pip install --upgrade setuptools wheel
RUN python setup.py bdist_wheel

# Install sd-scripts ===========================================
FROM python:3.10.6 as sd-scripts

# feat https://github.com/kohya-ss/sd-scripts/blob/main/README-ja.md

RUN git clone https://github.com/kohya-ss/sd-scripts.git
WORKDIR /sd-scripts
RUN pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip install --upgrade -r requirements.txt
COPY --from=xformers /xformers/dist/ /tmp/xformers/
RUN pip install -U -I --no-deps $(find /tmp/xformers/ -name *.whl)

RUN apt-get update \
  && apt-get install -y libgl1-mesa-dev

COPY --chmod=111 ./run.sh /sd-scripts/run.sh

ENTRYPOINT [ "/sd-scripts/run.sh" ]

CMD [ "" ]

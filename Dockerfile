FROM public.ecr.aws/lambda/python:3.11

# Instalar dependências do sistema (ffmpeg do static build)
RUN yum install -y gcc gcc-c++ tar xz && \
    curl -L https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz -o /tmp/ffmpeg.tar.xz && \
    tar -xf /tmp/ffmpeg.tar.xz -C /tmp && \
    mv /tmp/ffmpeg-*-amd64-static/ffmpeg /usr/local/bin/ && \
    mv /tmp/ffmpeg-*-amd64-static/ffprobe /usr/local/bin/ && \
    rm -rf /tmp/ffmpeg* && \
    yum clean all

# Copiar requirements e instalar
COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install --no-cache-dir -r requirements.txt

# Baixar recursos NLTK no diretório correto para Lambda
ENV NLTK_DATA=/var/task/nltk_data
ENV TRANSFORMERS_CACHE=/tmp/transformers_cache
ENV HF_HOME=/tmp/huggingface
ENV XDG_CACHE_HOME=/tmp/.cache
RUN mkdir -p ${NLTK_DATA} && \
    python -c "import nltk; \
    nltk.download('punkt', download_dir='/var/task/nltk_data'); \
    nltk.download('punkt_tab', download_dir='/var/task/nltk_data'); \
    nltk.download('stopwords', download_dir='/var/task/nltk_data'); \
    nltk.download('vader_lexicon', download_dir='/var/task/nltk_data')"

# Baixar modelos SpaCy via pip (compatível com spacy 3.7.2)
RUN pip install https://github.com/explosion/spacy-models/releases/download/pt_core_news_sm-3.7.0/pt_core_news_sm-3.7.0-py3-none-any.whl
RUN pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl

# Copiar código
COPY nlp/ ${LAMBDA_TASK_ROOT}/nlp/
COPY data/ ${LAMBDA_TASK_ROOT}/data/
COPY lambda_function.py ${LAMBDA_TASK_ROOT}

CMD ["lambda_function.lambda_handler"]

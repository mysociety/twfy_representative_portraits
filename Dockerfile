FROM python:3.10-bullseye

ENV DEBIAN_FRONTEND noninteractive
ENV PROJECT_FOLDER twfy_representative_portraits
COPY pyproject.toml poetry.loc[k] src /
RUN curl -sSL https://install.python-poetry.org | python - && \
    mkdir "src/$PROJECT_FOLDER" -p && \
    echo 'export PATH="/root/.local/bin:$PATH"' > ~/.bashrc && \
    export PATH="/root/.local/bin:$PATH"  && \
    poetry config virtualenvs.create false && \
    poetry self add poetry-bumpversion && \
    poetry install && \
    rmdir "src/$PROJECT_FOLDER" && \
    echo "/workspaces/$PROJECT_FOLDER/src/" > "/usr/local/lib/python3.10/site-packages/$PROJECT_FOLDER.pth"
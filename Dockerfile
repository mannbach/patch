FROM python:3.9-bullseye

ARG WORKDIR

# Specify workdir in image
WORKDIR /patch

# Copy local files to workdir
ADD . /patch/

# Install packages
RUN \
    pip install --upgrade pip &&\
    pip install -r requirements.txt &&\
    pip install -e /NetworkInequalities &&\
    pip install -e ./

# Print
CMD ["tail", "-f", "/dev/null"]
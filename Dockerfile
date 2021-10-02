FROM tensorflow/tensorflow:2.5.1-gpu

ENV DEBIAN_FRONTEND=noninteractive

# Apt installs
RUN apt-get update && apt-get install -y tree vim

# Install all the requirements
COPY requirements.txt /tmp
RUN python3 -m pip install --upgrade pip && cd /tmp && python3 -m pip install -r requirements.txt --user

# Copy model

COPY models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5 /root/.keras/models/

# Copy the project
COPY deeplab/ /home/deeplab
COPY *.py /home
#COPY server.py /home

# create output directory
RUN mkdir -p "/home/output"

# Expose ports
#EXPOSE 5000

# Set working directory
WORKDIR /home

# Startup command
#CMD ["python3", "server.py"]


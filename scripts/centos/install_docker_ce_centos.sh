#!/bin/bash
echo "Installing docker on centos"
sudo yum install -y yum-utils \
  device-mapper-persistent-data \
  lvm2

sudo yum-config-manager \
    --add-repo \
    https://download.docker.com/linux/centos/docker-ce.repo

echo "060A 61C5 1B55 8A7F 742B 77AA C52F EB6B 621E 9F35"
sudo yum install docker-ce docker-ce-cli containerd.io
sudo systemctl start docker
sudo docker run hello-world
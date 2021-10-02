echo "Install Docker"
sudo apt-get update
sudo apt-get install -y \
apt-transport-https \
ca-certificates \
curl \
software-properties-common

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo apt-key fingerprint 0EBFCD88
#9DC8 5822 9FC7 DD38 854A E2D8 8D81 803C 0EBF CD88
sudo add-apt-repository \
"deb [arch=amd64] https://download.docker.com/linux/ubuntu \
$(lsb_release -cs) \
stable"
sudo apt-get update
sudo apt-get install -y docker-ce

echo "Add user to the docker group"
sudo groupadd docker
sudo usermod -aG docker $USER
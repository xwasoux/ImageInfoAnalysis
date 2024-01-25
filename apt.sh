sudo apt update -y
sudo apt -y install \
    tig \
    htop \
    tmux \
    vim \
    tree

git config --global --add safe.directory /workspace/.git
echo "Enter your git account name: "
read user
echo "Enter your git account email: "
read email
git config --global user.name $user
git config --global user.email $email

git config core.fileMode false
sudo chmod -R 777 .
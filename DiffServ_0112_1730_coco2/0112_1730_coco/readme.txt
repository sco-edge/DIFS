# run the command with src as the destination for the generated files
# 1. generate the general Protobuf message classes (--cpp_out)
# 2. generate the gRPC service interface classes (--grpc_out)

protoc -I ./protos \
    --cpp_out=./src \
    --grpc_out=./src \
    --plugin=protoc-gen-grpc=`which grpc_cpp_plugin` \
    ./protos/query.proto ./protos/infaas_request_status.proto

[Rebuild client.cc]  
DiffServ/src$ modify client.cc
DiffServ/build$ cmake ..
DiffServ/build$ make -j$(nproc)

// for client.py
python -m grpc_tools.protoc \
    -I./protos \
    --python_out=./src \
    --grpc_python_out=./src \
    ./protos/query.proto ./protos/infaas_request_status.proto

// pyenv initial setup
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"

Apply the settings:
source ~/.bashrc  # if using Bash
source ~/.zshrc   # if using Zsh

// specify the version for the current directory
pyenv local 3.11.14

// create a virtual environment
python -m venv venv_diffserve

// activate the virtual environment
source venv_diffserve/bin/activate

[Installing pyenv]
install the required packages
sudo apt update
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
  libncurses5-dev libncursesw5-dev xz-utils tk-dev \
  libffi-dev liblzma-dev python3-openssl git

install pyenv
curl https://pyenv.run | bash

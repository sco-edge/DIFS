# 생성될 파일들이 저장될 위치를 src로 지정하여 명령 실행
# 1. 일반 Protobuf 메시지 클래스 생성 (--cpp_out)
# 2. gRPC 서비스 인터페이스 클래스 생성 (--grpc_out)

protoc -I ./protos \
    --cpp_out=./src \
    --grpc_out=./src \
    --plugin=protoc-gen-grpc=`which grpc_cpp_plugin` \
    ./protos/query.proto ./protos/infaas_request_status.proto

[client.cc 수정 빌드]  
DiffServ/src$ client.cc 수정
DiffServ/build$ cmake ..
DiffServ/build$ make -j$(nproc)

// client.py 용
python -m grpc_tools.protoc \
    -I./protos \
    --python_out=./src \
    --grpc_python_out=./src \
    ./protos/query.proto ./protos/infaas_request_status.proto

// pyenv 초기설정
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"

설정 적용:
source ~/.bashrc  # Bash 사용 시
source ~/.zshrc   # Zsh 사용 시

// 현재 디렉토리 버전 지정
pyenv local 3.11.14

// 가상환경 생성
python -m venv venv_diffserve

// 가상환경 활성화
source venv_diffserve/bin/activate

[pyenv 설치]
필요한 패키지 설치
sudo apt update
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
  libncurses5-dev libncursesw5-dev xz-utils tk-dev \
  libffi-dev liblzma-dev python3-openssl git

pyenv 설치
curl https://pyenv.run | bash

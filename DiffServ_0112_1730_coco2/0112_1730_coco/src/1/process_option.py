import argparse
import json
import sys
import os

def load_json(file_path):
    """JSON 파일을 로드하는 유틸리티 함수"""
    if not os.path.exists(file_path):
        print(f"Error: {file_path} 파일이 존재하지 않습니다.")
        return None
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_filename_by_id(data, target_id):
    """JSON 리스트에서 ID에 매칭되는 filename 반환"""
    for item in data:
        if item.get("id") == target_id:
            return item.get("filename")
    return None

def main():
    # 1. 인자 파싱 설정
    parser = argparse.ArgumentParser(description="Process AI Generation Options")
    
    parser.add_argument("-model", type=int, required=True, help="Model ID (from models.json)")
    parser.add_argument("-thread", type=int, default=1, help="Number of threads (n)")
    parser.add_argument("-port", type=int, default=50051, help="Port number")
    parser.add_argument("-sampler", type=int, required=True, help="Sampler ID (from samplers.json)")
    parser.add_argument("-npz", type=str, required=True, help="NPZ file path")

    args = parser.parse_args()

    # 2. JSON 데이터 로드
    models_data = load_json("models.json")
    samplers_data = load_json("samplers.json")

    if models_data is None or samplers_data is None:
        sys.exit(1)

    # 3. ID를 파일명으로 변환
    model_file = get_filename_by_id(models_data, args.model)
    sampler_name = get_filename_by_id(samplers_data, args.sampler)

    # 4. 결과 출력 및 유효성 검사
    print("=== 설정 확인 ===")
    if model_file:
        print(f"선택된 모델: {model_file} (ID: {args.model})")
    else:
        print(f"Error: ID {args.model}에 해당하는 모델을 찾을 수 없습니다.")

    if sampler_name:
        print(f"선택된 샘플러: {sampler_name} (ID: {args.sampler})")
    else:
        print(f"Error: ID {args.sampler}에 해당하는 샘플러를 찾을 수 없습니다.")

    print(f"쓰레드 수: {args.thread}")
    print(f"포트 번호: {args.port}")
    print(f"NPZ 경로: {args.npz}")
    print("================")

if __name__ == "__main__":
    main()
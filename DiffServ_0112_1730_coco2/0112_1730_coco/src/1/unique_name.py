import datetime
import secrets

now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
random_str = secrets.token_hex(3) # 6자리 랜덤 문자열
unique_name = f"{now}_{random_str}"
print(unique_name)
# 결과 예: 20251230_145125_7f3e1a
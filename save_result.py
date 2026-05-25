import base64
import json

# Postman 응답 전체를 response.json 파일로 저장했다고 가정
with open("response.json", "r", encoding="utf-8") as f:
    data = json.load(f)

image_b64 = data["image"]

with open("annotated_result.jpg", "wb") as f:
    f.write(base64.b64decode(image_b64))

print("annotated_result.jpg 저장 완료")
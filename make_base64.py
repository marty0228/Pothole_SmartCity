import base64

image_path = "test_model5.jpg"  # 테스트할 이미지 파일명으로 수정

with open(image_path, "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

with open("image_base64.txt", "w", encoding="utf-8") as f:
    f.write(image_b64)

print("image_base64.txt 저장 완료")
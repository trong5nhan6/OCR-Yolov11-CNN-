import zipfile
import os

zip_path = "data/icdar2003.zip"
extract_path = "data"

os.makedirs(extract_path, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("Giải nén thành công vào thư mục:", extract_path)

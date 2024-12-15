# alexnet-cv-team7
- Paper: https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf

# Cách cài dependencies web app:
* Cách 1: Sử dụng pip
1. Cài python 3.12
2. (Tùy chọn) python -m venv ./venv
3. (Tùy chọn) source venv/bin/activate
4. pip install -r requirements.txt

* Cách 2: Dùng conda
1. Cài miniconda hoặc anaconda https://docs.anaconda.com/miniconda/
2. conda env create -f environment.yml

# Cách chạy web app:
* Truy cập https://www.kaggle.com/code/hienvm/alexnet-pretrained-skin-diagnosis/output?scriptVersionId=211186752
* Tải các file output (không cần train augment)
* Ném các file này vào thư mục models
* Chạy app.py
* Vào trang http://127.0.0.1:5000

# Kết quả thí nghiệm
* Folder notebooks

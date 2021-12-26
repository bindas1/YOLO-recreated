# YOLO-recreated
Recreated YOLO algorithm for my engineering thesis.

University: Warsaw University of Technology

Supervisor: dr. inż Krystian Radlak

### How to run?
0. If conda is not installed install conda (for WSL: https://gist.github.com/kauffmanes/5e74916617f9993bc3479f401dfec7da)
1. Clone the repository and enter repository
```
git clone https://github.com/bindas1/YOLO-recreated.git
cd YOLO-recreated
```
2. Install conda environment and activate it:
```
conda env create -f ./requirements/conda-environment_linux.yaml
conda activate yolo
```
3. Add required packages
```
pip install -r ./requirements/requirements_linux.txt
```
4. Run
```
python main.py
```
When prompted choose 1 (if wandb account not created)
<img width="341" alt="image" src="https://user-images.githubusercontent.com/38891725/147409972-0ae85095-480d-4d42-92ff-3635280959af.png">

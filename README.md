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
4. Create wandb account
Go to https://wandb.ai/authorize?signup=true, sign in using github, google or create account. After the process is complete You should receive your API key (it can be also accessed in settings for account)
5.
Edit `train.py`
Change line 14 entity from `bindas1` to your username:
```
with wandb.init(project="YOLO-recreated", entity="bindas1", config=hyp):
```
6. Run
```
python main.py
```
When prompted choose 2 (if wandb account not created)

<img width="341" alt="image" src="https://user-images.githubusercontent.com/38891725/147409972-0ae85095-480d-4d42-92ff-3635280959af.png">

Copy the API from the sign up process is complete and paste to the terminal. (This process only needs to be done once)

After the login you should see that the data is being downloaded. If You want to make sure that the installation process is complete I suggest running the code with `config["is_one_batch"]` set to `True` in `main.py`.

You can access all the metrics in your profile, each run produces new instance on wandb page.

### FAQ
If you receive the RuntimeErrror: CUDA out of memory You should change the `batch_size` in `config` in `main.py` to smaller number.



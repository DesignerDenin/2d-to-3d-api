1. Install Python 3.10 (using brew here)
```
brew install python@3.10
```

&nbsp;

2. Create Python 3.10 Virtual Env
```
python3.10 -m venv .venv
```

&nbsp;

3. Install dependencies/packages
```
pip install -r requirements.txt
```

&nbsp;

4. Build MISE Extension module
```
python scripts/setup.py build_ext --inplace
```

&nbsp;

5. Try running the 3D Model Generation script (have atleast one image in `/ouputs` folder)
```
python generate.py configs/demo.yaml
```

&nbsp;

6. To run the API locally
```
flask run
```

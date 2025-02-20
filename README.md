# Project setup
- setup local venv for the project
    - `python -m venv venv` - creates a venv named venv
    - `.\venv\Scripts\activate` (windows) `source ./venv/bin/activate` (Mac/ Linux) - activates venv
        - you should see a (venv) in your command line prompt
    - depending on your IDE you may have to set the python interpreter to use the venv python executable (shell should be fine)
        - in vscode you can press control p and search `python interpreter` and change it appropriately
- install python dependencies from requirements.txt
    - `pip install -r requirements.txt`
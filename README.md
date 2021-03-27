# Slack bot for deploying data tables to production

## Setup
- Install conda (https://docs.conda.io/en/latest/)
- Initialize conda environment ```conda create -n ${PWD##*/} python=3.7.6```
- Activate your conda environment ```conda activate ${PWD##*/}```
- Install all dependencies ```pip install -r requirements.txt```
- Install spacy english core ```python -m spacy download en_core_web_sm```
- Train the bot ```python bot/train.py```
- Copy ".env-sample" to ".env"
- Fill in all environment variables. All slack related variables can be found 
under https://api.slack.com/apps/<id>/general.


## Run flask locally

- start ngrok ```ngrok http 8080```
- source our env variables ```source .env```
- start flask server ```python main-dev.py```
- copy ngrok public url
- update slack bot event subscription (https://api.slack.com/apps/<id>/event-subscriptions?)
- update slack bot interactivity (https://api.slack.com/apps/<id>/interactive-messages?)

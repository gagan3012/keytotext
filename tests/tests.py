from keytotext.pipeline import pipeline
import requests


def apitests():
    r = requests.get('http://127.0.0.1:5000')
    if r.status_code == 404:
        print("error")

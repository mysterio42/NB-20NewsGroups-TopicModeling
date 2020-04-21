import requests

url = 'http://0.0.0.0:5000/topic'
# payload = {'sentence': 'Software Engineering is getting hotter and hotter nowdays'}
# payload = {'sentence': 'The Effect of Intracoronary Infusion of Autologous Bone Marrow-Derived Lineage-Negative Stem/Progenitor Cells on Remodeling of Post-Infarcted Heart in Patient with Acute Myocardial Infarction'}
payload = {'sentence': 'They have encountered many arguments and much supposed evidence for the existence of God, but they have found all of it to be invalid or inconclusive.'}

if __name__ == '__main__':
    r = requests.post(url, json=payload)
    print(r.text)

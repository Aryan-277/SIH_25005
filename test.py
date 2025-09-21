import requests

url = "http://127.0.0.1:5000/assess"
files = {
    'side_img': open(r"C:\ATC_test1\dataset\images\side\1.png", 'rb'),
    'back_img': open(r"C:\ATC_test1\dataset\images\back\1.png", 'rb')
}
data = {
    'numeric_features': '{"Oblique body length (cm)":161,"Withers height(cm)":124,"Heart girth(cm)":190,"Hip length (cm)":47}'
}

response = requests.post(url, files=files, data=data)
print(response.json())

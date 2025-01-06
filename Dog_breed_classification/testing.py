import requests

# Test health check
response = requests.get('http://localhost:5000/health')
print("Health Check:", response.json())

# Test prediction
image_path = './dog2.jpg'  # Ganti dengan path gambar
files = {'file': open(image_path, 'rb')}
response = requests.post('http://localhost:5000/predict', files=files)
print("\nPrediction:", response.json())

# Test get classes
response = requests.get('http://localhost:5000/classes')
print("\nClasses:", response.json())
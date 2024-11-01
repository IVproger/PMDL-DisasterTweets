import requests

def health_check():
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            exit(0)
        else:
            exit(1)
    except requests.exceptions.RequestException:
        exit(1)

if __name__ == "__main__":
    health_check()
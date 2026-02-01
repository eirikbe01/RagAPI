import requests

def test_kubernetes_query():
    response = requests.post("http://127.0.0.1:8000/query?q=What is Kubernetes?")
    
    if response.status_code != 200:
        raise Exception(f"Server returned {response.status_code}: {response.text}")
    
    answer = response.json()["answer"]

    # Check for key concepts
    assert "orchestration" in answer.lower(), "Missing 'orchestration' keyword"
    assert "container" in answer.lower(), "Missing 'container' keyword"
    
    print("✅ Kubernetes query test passed")

def test_docker_query():
    response = requests.post("http://127.0.0.1:8000/query?q=What is Docker?")
    
    if response.status_code != 200:
        raise Exception(f"Server returned {response.status_code}: {response.text}")
    
    answer = response.json()["answer"]

    # Check for key concepts
    assert "packaging" in answer.lower(), "Missing 'packaging' keyword"
    assert "container" in answer.lower(), "Missing 'container' keyword"
    
    print("✅ Docker query test passed")

if __name__ == "__main__":
    test_kubernetes_query()
    test_docker_query()
    print("All semantic tests passed!")

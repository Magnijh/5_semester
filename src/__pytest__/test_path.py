#All test cases for testing different paths
def test_root(client): 
    res = client.get("/")
    assert res.status_code == 200

def test_info(client): 
    res = client.get("/info")
    assert res.status_code == 200

def test_readMe(client): 
    res = client.get("/readme")
    assert res.status_code == 200

def test_fields(client): 
    res = client.get("/fields")
    assert res.status_code == 200
    
def test_render(client): 
    res = client.get("/render")
    assert res.status_code == 200

def test_data(client): 
    res = client.get("/data")
    assert res.status_code == 200
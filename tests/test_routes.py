def test_core_pages_render(client):
    for path in ["/", "/train", "/chat", "/models", "/about"]:
        response = client.get(path)
        assert response.status_code == 200


def test_chat_api_rejects_bad_model_name(client):
    response = client.post("/chat_api", json={"model_name": "../bad", "message": "hi"})
    data = response.get_json()
    assert data["success"] is False

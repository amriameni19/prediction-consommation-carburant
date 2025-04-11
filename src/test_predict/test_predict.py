from predict_model import predict

def test_predict_model():
    sample_features = {
        'weight': 3000,
        'acceleration': 12.0,
        'displacement': 200.0,
        'cylinders': 4,
        'model_year': 76,
        'horsepower': 90.0
    }

    result = predict(sample_features)

    assert 'consommation_reelle' in result
    assert isinstance(result['consommation_reelle'], float)

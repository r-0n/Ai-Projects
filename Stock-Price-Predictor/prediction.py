def predict(model, scaler, input_data):
    # Scale the input data
    input_data_scaled = scaler.transform([input_data])
    
    # Make prediction
    prediction = model.predict(input_data_scaled)
    
    return prediction[0]

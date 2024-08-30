def predict_text(text, model, vectorizer):
    text_vect = vectorizer.transform([text])
    prediction = model.predict(text_vect)
    sentiment = "positive" if prediction[0] == 1 else "negative"
    return sentiment

def predict_file(text_series, model, vectorizer):
    text_vect = vectorizer.transform(text_series)
    predictions = model.predict(text_vect)
    results = [{"text": text, "sentiment": "positive" if pred == 1 else "negative"} for text, pred in zip(text_series, predictions)]
    return results

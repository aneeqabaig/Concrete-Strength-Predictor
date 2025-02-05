from django.shortcuts import render
import joblib
import numpy as np

# Load the trained model
model = joblib.load("cement_strength_model.pkl")

def predict_strength(request):
    result = None

    if request.method == "POST":
        days = request.POST.get("days")
        wc_ratio = request.POST.get("wc_ratio")

        if days and wc_ratio:
            try:
                days = float(days)
                wc_ratio = float(wc_ratio)

                # Prepare input for prediction
                input_data = np.array([[days, wc_ratio]])

                # Predict using the model
                result = model.predict(input_data)[0]

                result = round(result, 2)  # Round to 2 decimal places
            except ValueError:
                result = "Invalid input. Please enter numbers only."

    return render(request, "predict.html", {"result": result})

from django.shortcuts import render, redirect
from django.contrib import messages
from mainapp.models import *
from adminapp.models import *
from django.utils import timezone
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pytz
from datetime import datetime
from django.core.files.storage import default_storage

from django.conf import settings

# Create your views here.

def user_dashboard(req):
    user_id = req.session["user_id"]
    user = UserModel.objects.get(user_id=user_id)
    Feedbacks_users_count = Feedback.objects.all().count()
    all_users_count = UserModel.objects.all().count()

    if user.Last_Login_Time is None:
        IST = pytz.timezone("Asia/Kolkata")
        current_time_ist = datetime.now(IST).time()
        user.Last_Login_Time = current_time_ist
        user.save()
        return redirect("user_dashboard")

    return render(
        req,
        "user/user-dashboard.html",
        {
            # "predictions": prediction_count,
            "user_name": user.user_name,  
            "feedback_count": Feedbacks_users_count,
            "all_users_count": all_users_count,
        },
    )

    
def profile(req):
    user_id = req.session.get("user_id")
    if not user_id:
        messages.error(req, "User not logged in.")
        return redirect("login")

    user = UserModel.objects.get(user_id=user_id)

    if req.method == "POST":
        user.user_name = req.POST.get("username")
        user.user_age = req.POST.get("age")
        user.user_address = req.POST.get("address")
        user.user_contact = req.POST.get("mobile_number")
        user.user_email = req.POST.get("email")
        user.user_password= req.POST.get("password")
        # Handle image upload if present
        if 'profilepic' in req.FILES:
            user.user_image = req.FILES['profilepic']
        user.save()
        messages.success(req, "Profile updated successfully.")
        return redirect("profile")

    context = {"i": user}
    return render(req, "user/profile.html",context)



def user_feedback(req):
    id = req.session["user_id"]
    uusser = UserModel.objects.get(user_id=id)
    if req.method == "POST":
        rating = req.POST.get("rating")
        review = req.POST.get("review")
        if not rating or not review:
            messages.warning(req, "Enter all the fields to continue!")
            return render (req, "user/user-feedback.html")
        sid = SentimentIntensityAnalyzer()
        score = sid.polarity_scores(review)
        sentiment = None
        if score["compound"] > 0 and score["compound"] <= 0.5:
            sentiment = "positive"
        elif score["compound"] >= 0.5:
            sentiment = "very positive"
        elif score["compound"] < -0.5:
            sentiment = "negative"
        elif score["compound"] < 0 and score["compound"] >= -0.5:
            sentiment = " very negative"
        else:
            sentiment = "neutral"
        Feedback.objects.create(
            Rating=rating, Review=review, Sentiment=sentiment, Reviewer=uusser
        )
        messages.success(req, "Feedback recorded")
        return redirect("user_feedback")
    return render(req, "user/user-feedback.html")

def user_logout(req):
    if "user_id" in req.session:
        view_id = req.session["user_id"]
        try:
            user = UserModel.objects.get(user_id=view_id)
            user.Last_Login_Time = timezone.now().time()
            user.Last_Login_Date = timezone.now().date()
            user.save()
            messages.info(req, "You are logged out.")
        except UserModel.DoesNotExist:
            pass
    req.session.flush()
    return redirect("login")

# -------------------------------------------------------

import numpy as np
import pickle
from django.shortcuts import render
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras.losses import KLDivergence, MeanSquaredError
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import VarianceScaling
import tensorflow.keras.backend as K
import os

# ✅ Define the ClusteringLayer class again
class ClusteringLayer(Layer):
    def __init__(self, n_clusters, **kwargs):
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters

    def build(self, input_shape):
        self.clusters = self.add_weight(name='clusters',
                                        shape=(self.n_clusters, input_shape[1]),
                                        initializer=VarianceScaling(scale=1.0, mode='fan_in', distribution='normal'),
                                        trainable=True)

    def call(self, inputs, **kwargs):
        q = 1.0 / (1.0 + K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2))
        q /= K.sum(q, axis=1, keepdims=True)
        return q

# ✅ Load the TF-IDF vectorizer
base_path = "C:\\Users\\a\\Codebook\\doctor-note-detection"
file_name = "Clinical\\vectorizer.pkl"
path = os.path.join(base_path, file_name)

with open(path, "rb") as f:
    vectorizer = pickle.load(f)

# ✅ Custom objects for model loading
custom_objects = {
    "ClusteringLayer": ClusteringLayer,
    "kld": KLDivergence(),
    "mse": MeanSquaredError()
}

# ✅ Load the DEC2Vec model with custom objects
model_path = os.path.join(base_path, "Clinical", "DEC2Vec_model.h5")

# Use custom_object_scope to properly load the model
with custom_object_scope(custom_objects):
    dec2vec_model = load_model(model_path)

print("Model loaded successfully!")

# ✅ Class label mapping
label_mapping = {
    0: "Deterioration",
    1: "ICU transfer",
    2: "Improvement",
    3: "Stable"
}

# ✅ Preprocessing function
def preprocess_text(text):
    """Preprocesses text using the TF-IDF vectorizer with consistent shape."""
    text = [text]  
    vector = vectorizer.transform(text).toarray()

    # Ensure the vector shape matches the model's input shape
    target_shape = dec2vec_model.input_shape[1]
    
    if vector.shape[1] != target_shape:
        padded_vector = np.zeros((1, target_shape))
        padded_vector[0, :vector.shape[1]] = vector
        vector = padded_vector

    return vector

# ✅ Prediction function
def predict_outcome(note, diagnosis):
    """Predicts the outcome for the given doctor's note and diagnosis."""
    
    # Combine inputs
    combined_text = f"Note: {note}. Diagnosis: {diagnosis}"
    input_vector = preprocess_text(combined_text)

    # Make predictions
    predictions = dec2vec_model.predict(input_vector)

    # Handle single-output model prediction
    if isinstance(predictions, list):  
        predictions = predictions[1]  # For multi-output models
    else:
        predictions = predictions  # For single-output models

    # Get the predicted cluster index
    predicted_cluster = np.argmax(predictions, axis=-1)[0]

    # Map the cluster index to the label
    predicted_label = label_mapping.get(predicted_cluster, "Unknown")
    
    return predicted_label

# ✅ Views

from django.urls import reverse

# Renders the input form and handles the form submission
def detection(req):
    """
    Displays the form and handles form submission.
    """
    if req.method == "POST":
        doctor_note = req.POST.get("doctor_note")
        diagnosis = req.POST.get("diagnosis")

        # Check if both fields are filled
        if not doctor_note or not diagnosis:
            return render(req, "user/detection.html", {"error": "Both fields are required!"})

        # Make the prediction
        predicted_outcome = predict_outcome(doctor_note, diagnosis)

        # Map the outcome label to CSS classes for styling
        css_class = {
            "Deterioration": "deteriorating",
            "ICU transfer": "icu-transfer",
            "Improvement": "improvement",
            "Stable": "stable"
        }.get(predicted_outcome, "unknown")

        # Store the data in the session
        req.session["context"] = {
            "doctor_note": doctor_note,
            "diagnosis": diagnosis,
            "predicted_outcome": predicted_outcome,
            "css_class": css_class
        }

        # Redirect to the result page
        return redirect(reverse("detection_result"))

    # Render the form if accessed through GET
    return render(req, "user/detection.html")


# Handles the form submission and displays the result
def detection_result(req):
    """
    Displays the prediction result page.
    """
    # Retrieve the context data from the session
    context = req.session.pop("context", None)

    if not context:
        # Redirect back to the form if accessed directly without form submission
        return redirect(reverse("detection"))

    # Render the result page with the prediction data
    return render(req, "user/detection-result.html", context)



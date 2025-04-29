from django.contrib import messages
from django.shortcuts import render, redirect, get_object_or_404
from django.core.paginator import Paginator
from django.utils.timezone import localtime
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
import numpy as np
from django.utils import timezone
import datetime
# Local imports
from mainapp.models import UserModel
from mainapp.models import Feedback
from adminapp.models import (
    UploadDatasetModel,
    MultinomialNBModel,
    DecisionTreeModel,
    LogisticRegressionModel,
    NaiveBayesModel
)
from adminapp.models import UploadDatasetModel, DecisionTreeModel, LogisticRegressionModel, RandomForestModel, NaiveBayesModel, MultinomialNBModel



# Create your views here.
def admin_dash(req):
    # Fetch the last 4 recent users ordered by 'date_time' in descending order
    recent_users = UserModel.objects.all().order_by('-Date_Time')[:4]

    # Count statistics
    all_users_count = UserModel.objects.all().count()
    pending_users_count = UserModel.objects.filter(User_Status="pending").count()
    rejected_users_count = UserModel.objects.filter(User_Status="removed").count()
    accepted_users_count = UserModel.objects.filter(User_Status="accepted").count()
    feedbacks_users_count = Feedback.objects.all().count()
    classification_count = UserModel.objects.all().count()

    # Format the date and time for each user
    formatted_users = []
    for user in recent_users:
        formatted_users.append({
            'user_image': user.user_image,
            'user_name': user.user_name,
            'user_email': user.user_email,
            'date_time': localtime(user.Date_Time).strftime('%H:%M %d/%m/%Y')
        })

    return render(
        req,
        "admin/admin-dashboard.html",
        {
            "all_users": all_users_count,
            "pending_users": pending_users_count,
            "rejected_users": rejected_users_count,
            "accepted_users": accepted_users_count,
            "recent_users": formatted_users,
            "feedback_count": feedbacks_users_count,
            "classification_count": classification_count,
        },
    )


def pendingusers(req):
    pending = UserModel.objects.filter(User_Status="pending")
    paginator = Paginator(pending, 5)
    page_number = req.GET.get("page")
    post = paginator.get_page(page_number)
    return render(req, "admin/pending-users.html", {"user": post})


import json

def graphcomparison(request):
    # Retrieve accuracy values from the database for all algorithms
    Multinomial_nb_details = MultinomialNBModel.objects.last()
    NaiveBayes_details = NaiveBayesModel.objects.last()
    Logistic_Regression = LogisticRegressionModel.objects.last()
    decision_details = DecisionTreeModel.objects.last()  
    Randomforest_details = RandomForestModel.objects.last()

    # Check if any model details are None
    if not all([Multinomial_nb_details, NaiveBayes_details, Logistic_Regression, decision_details, Randomforest_details]):
        messages.error(request, 'Run the Algorithms First.')
        return redirect('decision_tree')  

    context = {
        'multinomial_nb_accuracy': float(Multinomial_nb_details.Accuracy.strip('%')) if Multinomial_nb_details else 0,
        'naive_bayes_accuracy': float(NaiveBayes_details.Accuracy.strip('%')) if NaiveBayes_details else 0,
        'logistic_accuracy': float(Logistic_Regression.Accuracy.strip('%')) if Logistic_Regression else 0,
        'decision_accuracy': float(decision_details.Accuracy.strip('%')) if decision_details else 0,
        'random_forest_accuracy': float(Randomforest_details.Accuracy.strip('%')) if Randomforest_details else 0,
    }

    # Convert context dictionary to a JSON string
    context_json = json.dumps(context)

    return render(request, "admin/graph-comparison.html", {'context_json': context_json})




def adminlogout(req):
    messages.info(req, "You are logged out.")
    return redirect("adminlogin")

def allusers(req):
    all_users = UserModel.objects.all()
    paginator = Paginator(all_users, 5)
    page_number = req.GET.get("page")
    post = paginator.get_page(page_number)
    return render(req, "admin/all-users.html", {"allu": all_users, "user": post})


def delete_user(req, user_id):
    try:
        user = UserModel.objects.get(user_id=user_id)
        user.delete()
        messages.warning(req, "User was deleted successfully!")
    except UserModel.DoesNotExist:
        messages.error(req, "User does not exist.")
    except Exception as e:
        messages.error(req, f"An error occurred: {str(e)}")
    
    return redirect("allusers")


# Acept users button
def accept_user(req, id):
    try:
        status_update = UserModel.objects.get(user_id=id)
        status_update.User_Status = "accepted"
        status_update.save()
        messages.success(req, "User was accepted successfully!")
    except UserModel.DoesNotExist:
        messages.error(req, "User does not exist.")
    except Exception as e:
        messages.error(req, f"An error occurred: {str(e)}")
    
    return redirect("pendingusers")


# Remove user button
def reject_user(req, id):
    status_update2 = UserModel.objects.get(user_id=id)
    status_update2.User_Status = "removed"
    status_update2.save()
    messages.warning(req, "User was Rejected..!")
    return redirect("pendingusers")

# Change status users button
def change_status(req, id):
    user_data = UserModel.objects.get(user_id=id)
    if user_data.User_Status == "removed":
        user_data.User_Status = "accepted"
        user_data.save()
    elif user_data.User_Status == "accepted":
        user_data.User_Status = "removed"
        user_data.save()
    elif user_data.User_Status == "pending":
        messages.info(req, "Accept the user first..!")
        return redirect ("allusers")
    messages.success(req, "User status was changed..!")
    return redirect("allusers")

def feedback(req):
    feed = Feedback.objects.all()
    return render(req, "admin/feedback.html", {"back": feed})


def sentimentanalysis(req):
    fee = Feedback.objects.all()
    return render(req, "admin/sentiment-analysis.html", {"cat": fee})


def sentimentanalysisgraph(req):
    positive = Feedback.objects.filter(Sentiment="positive").count()
    very_positive = Feedback.objects.filter(Sentiment="very positive").count()
    negative = Feedback.objects.filter(Sentiment="negative").count()
    very_negative = Feedback.objects.filter(Sentiment="very negative").count()
    neutral = Feedback.objects.filter(Sentiment="neutral").count()
    context = {
        "vp": very_positive,
        "p": positive,
        "neg": negative,
        "vn": very_negative,
        "ne": neutral,
    }
    return render(req, "admin/sentiment-analysis-graph.html", context)



def create_histogram(df, ax):
    if 'amount' in df.columns:
        ax.hist(df['amount'].dropna(), bins=20, color='skyblue', edgecolor='black')
        ax.set_title('Histogram of Amount')
    else:
        ax.set_title('Column "Amount" not found')

def create_boxplot(df, ax):
    if 'oldbalanceOrg' in df.columns:
        sns.boxplot(df['oldbalanceOrg'].dropna(), color='green', ax=ax)
        ax.set_title('Boxplot of OldBalanceOrig')
        ax.set_xlabel('OldBalanceOrig')
    else:
        ax.set_title('Column "OldBalanceOrig" not found')

def create_lineplot(df, ax):
    if 'newbalanceOrig' in df.columns and 'oldbalanceOrg' in df.columns:
        sns.lineplot(x='newbalanceOrig', y='oldbalanceOrg', data=df, ax=ax)
        ax.set_title('Line Plot of oldbalanceOrg vs. newbalanceOrig')
        ax.set_xlabel('newbalanceOrig')
        ax.set_ylabel('oldbalanceOrg')
    else:
        ax.set_title('Required columns "newbalanceOrig" or "oldbalanceOrg" not found')

def create_countplot(df, ax):
    sns.countplot(x=df['Outcome'], data=df, ax=ax)
    ax.set_title('Count Chart')
    ax.set_xlabel('Outcome')

def plot_to_base64(fig):
    img_buffer = BytesIO()
    fig.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    img_data = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    plt.close(fig) 
    return f'data:image/png;base64,{img_data}'

def dataexploration(request):
    if not UploadDatasetModel.objects.exists():
        messages.error(request, 'Upload Dataset First.')
        return render(request, 'admin/admin-upload.html', {})

    dataset = UploadDatasetModel.objects.last()
    try:
        df = pd.read_csv(dataset.dataset.path)
    except Exception as e:
        messages.error(request, f'Error reading dataset: {e}')
        return render(request, 'admin/admin-upload.html', {})

    # Sort the dataset in descending order based on 'Amount' (or another column)
    df_sorted = df.sort_values(by='amount', ascending=False)  # Change column if needed

    # Take the top 1% of the sorted dataset
    df_sample = df_sorted.head(int(len(df) * 0.01))  # Adjust percentage if needed

    # Create subplots with 2 rows and 2 columns
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))

    # Plot the Histogram
    create_histogram(df_sample, axes[0, 0])

    # Plot the Boxplot
    create_boxplot(df_sample, axes[0, 1])

    # Plot the Line Plot
    create_lineplot(df_sample, axes[1, 0])

    # Plot the Countplot
    create_countplot(df_sample, axes[1, 1])

    # Convert the entire figure to base64-encoded image for rendering in HTML
    figure_img = plot_to_base64(fig)

    # Close the figure to free up resources
    plt.close(fig)

    messages.success(request, 'Data Exploration Analysis Completed Successfully.')
    return render(request, 'admin/data-exploration.html', {
        'figure_img': figure_img,
    })



# ------------------- Helper Functions ------------------- #
import pandas as pd
from django.shortcuts import render, redirect
from django.contrib import messages
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, classification_report)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler

# Import your Django models appropriately:
# from .models import DecisionTreeModel, RandomForestModel, NaiveBayesModel, MultinomialNBModel, LogisticRegressionModel, UploadDatasetModel

def load_and_prepare_data():
    try:
        data = pd.read_csv('media/dataset/cleaned_data.csv')
    except Exception as e:
        return None, None, None, None, None, f"Error reading CSV file: {e}"
    
    target_column = 'Outcome'
    if target_column not in data.columns:
        return None, None, None, None, None, f"Target column '{target_column}' not found in the dataset."
    
    # Separate features and target
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    
    # Convert categorical features in X to dummy variables (if any)
    X = pd.get_dummies(X, drop_first=True)
    
    # Save the feature column names for further analysis if needed
    feature_columns = list(X.columns)
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Return six values: X_train, X_test, y_train, y_test, feature_columns, and error (None)
    return X_train, X_test, y_train, y_test, feature_columns, None

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Train and evaluate the model using test metrics only."""
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_test_pred) * 100
    precision = precision_score(y_test, y_test_pred, average='weighted') * 100
    recall = recall_score(y_test, y_test_pred, average='weighted') * 100
    f1 = f1_score(y_test, y_test_pred, average='weighted') * 100
    report = classification_report(y_test, y_test_pred)
    
    # Return exactly five values: accuracy, precision, recall, f1, and report
    return accuracy, precision, recall, f1, report



# ------------------- Algorithms ------------------- #
def multinomial_nb(request):
    return render(request,'admin/multinomial-nb.html')

from sklearn.preprocessing import MinMaxScaler

def multinomial_nb_result(request):
    X_train, X_test, y_train, y_test, feature_columns, error = load_and_prepare_data()
    if error:
        messages.error(request, error)
        return redirect('admin_upload')
    
    # Re-scale X_train and X_test using MinMaxScaler to ensure all features are non-negative
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = MultinomialNB()
    accuracy, precision, recall, f1, report = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    MultinomialNBModel.objects.create(
        Name="MultinomialNB",
        Accuracy=f"{accuracy:.2f}",
        Precision=f"{precision:.2f}",
        Recall=f"{recall:.2f}",
        F1_Score=f"{f1:.2f}"
    )
    
    messages.success(request, 'MultinomialNB algorithm executed successfully.')
    return render(request, 'admin/multinomial-nb-result.html', {
        'name': "MultinomialNB",
        'accuracy': f"{accuracy:.2f}",
        'precision': f"{precision:.2f}",
        'recall': f"{recall:.2f}",
        'f1': f"{f1:.2f}",
        'report': report
    })

def decision_tree(request):
    return render(request,'admin/decision-tree.html')

def decision_tree_result(request):
    X_train, X_test, y_train, y_test, feature_columns, error = load_and_prepare_data()
    if error:
        messages.error(request, error)
        return redirect('admin_upload')
    
    model = DecisionTreeClassifier(random_state=42)
    accuracy, precision, recall, f1, report = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    DecisionTreeModel.objects.create(
        Name="Decision Tree",
        Accuracy=f"{accuracy:.2f}",
        Precision=f"{precision:.2f}",
        Recall=f"{recall:.2f}",
        F1_Score=f"{f1:.2f}"
    )
    
    messages.success(request, 'Decision Tree algorithm executed successfully.')
    return render(request, 'admin/decision-tree-result.html', {
        'name': "Decision Tree",
        'accuracy': f"{accuracy:.2f}",
        'precision': f"{precision:.2f}",
        'recall': f"{recall:.2f}",
        'f1': f"{f1:.2f}",
        'report': report
    })


def random_forest(request):
    return render(request,'admin/random-forest.html')

def random_forest_result(request):
    X_train, X_test, y_train, y_test, feature_columns, error = load_and_prepare_data()
    if error:
        messages.error(request, error)
        return redirect('admin_upload')
    
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    accuracy, precision, recall, f1, report = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    RandomForestModel.objects.create(
        Name="Random Forest",
        Accuracy=f"{accuracy:.2f}",
        Precision=f"{precision:.2f}",
        Recall=f"{recall:.2f}",
        F1_Score=f"{f1:.2f}"
    )
    
    messages.success(request, 'Random Forest algorithm executed successfully.')
    return render(request, 'admin/random-forest-result.html', {
        'name': "Random Forest",
        'accuracy': f"{accuracy:.2f}",
        'precision': f"{precision:.2f}",
        'recall': f"{recall:.2f}",
        'f1': f"{f1:.2f}",
        'report': report
    })

def logistic_regression(request):
    return render(request,'admin/logistic-regression.html')

def logistic_regression_result(req):
    # Check if results already exist in LogisticRegressionModel
    latest_algo = LogisticRegressionModel.objects.last()
    if latest_algo:
        messages.success(req, 'Successfully fetched Logistic Regression results.')
        return render(req, 'admin/logistic-regression-result.html', {'i': latest_algo})
    
    # Retrieve the latest uploaded dataset
    data = UploadDatasetModel.objects.last()
    if data is None:
        messages.error(req, 'No dataset available.')
        return render(req, "admin/upload-dataset.html")

    file = str(data.dataset)
    df = pd.read_csv(f'./media/{file}')
    
    # Prepare the data
    # Drop the target column from features and convert categorical features to numeric
    X = df.drop('Outcome', axis=1)
    X = pd.get_dummies(X, drop_first=True)  # This converts 'Female', 'Male', etc. to numeric columns
    y = df['Outcome']  # Keep target as is (string labels are acceptable in sklearn classification)
    
    # Handle class imbalance with optional oversampling
    use_oversampling = True  # Enable oversampling for balanced class distribution
    if use_oversampling:
        rs = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = rs.fit_resample(X, y)
    else:
        X_resampled, y_resampled = X, y

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, 
        test_size=0.2, 
        random_state=42
    )

    # Initialize and train the Logistic Regression model
    lr_model = LogisticRegression(random_state=42, solver='liblinear')
    lr_model.fit(X_train, y_train)

    # Perform cross-validation on training data only
    cv_scores = cross_val_score(lr_model, X_train, y_train, cv=5, scoring='accuracy')
    print("Cross-Validation Scores:", cv_scores)
    print("Mean Cross-Validation Accuracy:", cv_scores.mean() * 100)

    # Make predictions on the test set
    y_pred = lr_model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred, average='weighted') * 100
    recall = recall_score(y_test, y_pred, average='weighted') * 100
    f1 = f1_score(y_test, y_pred, average='weighted') * 100

    print("Logistic Regression Model Metrics:")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall: {recall:.2f}%")
    print(f"F1 Score: {f1:.2f}%")
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Save results to the database
    LogisticRegressionModel.objects.create(
        Name="Logistic Regression Algorithm",
        Accuracy=f"{accuracy:.2f}",
        Precision=f"{precision:.2f}",
        Recall=f"{recall:.2f}",
        F1_Score=f"{f1:.2f}"
    )
    print(accuracy, precision, recall, f1)
    messages.success(req, 'Logistic Regression algorithm executed successfully.')
    i = {'Name': "Logistic Regression", 'Accuracy': f"{accuracy:.2f}", 'Precision': f"{precision:.2f}", 'Recall': f"{recall:.2f}", 'F1': f"{f1:.2f}", 'Report': classification_report(y_test, y_pred)}
    return render(req, 'admin/logistic-regression-result.html', {'i': i})


def naive_bayes(request):
    return render(request,'admin/naive-bayes.html')

def naive_bayes_result(request):
    X_train, X_test, y_train, y_test, feature_columns, error = load_and_prepare_data()
    if error:
        messages.error(request, error)
        return redirect('admin_upload')
    
    model = GaussianNB()
    accuracy, precision, recall, f1, report = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    NaiveBayesModel.objects.create(
        Name="Naive Bayes",
        Accuracy=f"{accuracy:.2f}",
        Precision=f"{precision:.2f}",
        Recall=f"{recall:.2f}",
        F1_Score=f"{f1:.2f}"
    )
    
    messages.success(request, 'Naive Bayes algorithm executed successfully.')
    return render(request, 'admin/naive-bayes-result.html', {
        'name': "Naive Bayes",
        'accuracy': f"{accuracy:.2f}",
        'precision': f"{precision:.2f}",
        'recall': f"{recall:.2f}",
        'f1': f"{f1:.2f}",
        'report': report
    })


# ----------------------------------------------------------------------

def adminupload(request):
    if request.method == 'POST':
        file = request.FILES['data_file']
        file_size = file.size  # File size is already in bytes

        # Ensure user is included (assuming user is logged in)
        UploadDatasetModel.objects.create(
            file_size=file_size,  # Store file size in bytes
            dataset=file
        )

        messages.success(request, 'Your dataset was uploaded successfully.')
    
    return render(request, "admin/admin-upload.html")


def delete_dataset(request, id):
    try:
        dataset = get_object_or_404(UploadDatasetModel, id=id)  # Fetch dataset safely
        dataset.delete()
        messages.warning(request, 'Dataset was deleted..!')
    except Exception as e:
        messages.error(request, f"Error deleting dataset: {e}")
    
    return redirect('adminview')

def adminview(request):
    dataset = UploadDatasetModel.objects.all()  # Get all uploaded datasets
    paginator = Paginator(dataset, 5)  # Paginate, 5 items per page
    page_number = request.GET.get('page')
    post = paginator.get_page(page_number)

    # Convert UTC to local time before passing to template
    for data in dataset:
        data.uploaded_at = timezone.localtime(data.uploaded_at)
    return render(request, "admin/admin-view.html", {'data': post})  # Pass paginated data


def adminviewresult(request):
    data = UploadDatasetModel.objects.last()  # Get the latest uploaded dataset
    
    if not data:
        messages.error(request, "No dataset found.")
        return redirect('adminview')

    file_path = data.dataset.path  # Correctly get the file path
    
    try:
        df = pd.read_csv(file_path, nrows=50)  # Load only the first 50 rows
        table = df.to_html(classes="table table-bordered", table_id="data_table")  # Add styling for better UI
    except Exception as e:
        messages.error(request, f"Error reading file: {e}")
        return redirect('adminview')

    return render(request, "admin/admin-view-result.html", {'t': table})



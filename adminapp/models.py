from django.db import models
from djongo import models
from django.contrib.auth.models import User


# Create your models here.
class UploadDatasetModel(models.Model):
    s_no = models.AutoField(primary_key = True)   # MongoDB's default primary key
    dataset = models.FileField(upload_to='dataset/')  
    file_size = models.PositiveIntegerField()
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'upload_dataset'



class LogisticRegressionModel(models.Model):
    S_NO = models.AutoField(primary_key=True)
    Accuracy = models.CharField(max_length=100)
    Precision = models.CharField(max_length=100)
    F1_Score = models.CharField(max_length=100)
    Recall = models.CharField(max_length=100)
    Name = models.CharField(max_length=100)
    
    class Meta:
        db_table = 'Logistic_Regression'


class DecisionTreeModel(models.Model):
    S_NO = models.AutoField(primary_key=True)
    Accuracy = models.CharField(max_length=100)
    Precision = models.CharField(max_length=100)
    F1_Score = models.CharField(max_length=100)
    Recall = models.CharField(max_length=100)
    Name = models.CharField(max_length=100)
    
    class Meta:
        db_table = 'Decision_Tree'


class RandomForestModel(models.Model):
    S_NO = models.AutoField(primary_key=True)
    Accuracy = models.CharField(max_length=100)
    Precision = models.CharField(max_length=100)
    F1_Score = models.CharField(max_length=100)
    Recall = models.CharField(max_length=100)
    Name = models.CharField(max_length=100)
    
    class Meta:
        db_table = 'Random_Forest'


class MultinomialNBModel(models.Model):
    S_NO = models.AutoField(primary_key=True)
    Accuracy = models.CharField(max_length=100)
    Precision = models.CharField(max_length=100)
    F1_Score = models.CharField(max_length=100)
    Recall = models.CharField(max_length=100)
    Name = models.CharField(max_length=100)
    
    class Meta:
        db_table = 'Multinomial_NB'


class NaiveBayesModel(models.Model):
    S_NO = models.AutoField(primary_key=True)
    Accuracy = models.CharField(max_length=100)
    Precision = models.CharField(max_length=100)
    F1_Score = models.CharField(max_length=100)
    Recall = models.CharField(max_length=100)
    Name = models.CharField(max_length=100)
    
    class Meta:
        db_table = 'Naive_Bayes'



class ComparisonGraphModel(models.Model):
    S_No = models.AutoField(primary_key=True)
    Cnn = models.CharField(max_length=10, null=True)
    MobileNet = models.CharField(max_length=10, null=True)
    Densenet = models.CharField(max_length=10, null=True)

    class Meta:
        db_table = "Comparisongraph"



class Train_test_split_model(models.Model):
    S_No = models.AutoField(primary_key=True)
    Images_training = models.CharField(max_length=10, null=True)
    Images_validation = models.CharField(max_length=10, null=True)
    Images_testing = models.CharField(max_length=10, null=True)
    Images_classes = models.CharField(max_length=10, null=True)

    class Meta:
        db_table = "Traintestsplit"

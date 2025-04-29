"""
URL configuration for doctor_note project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

from userapp import views as user_views
from adminapp import views as admin_views
from mainapp import views as main_views

from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path("", main_views.index, name="index"),
    path("index", main_views.index, name="index"),
    path("about",main_views.about, name="about"),
    path("contact",main_views.contact, name="contact"),
    path("login",main_views.login, name="login"),
    path("otp",main_views.otp, name="otp"),
    path("register",main_views.signup, name="signup"),
    path("admin-login",main_views.adminlogin, name="adminlogin"),
    # User
    path("user-dashboard", user_views.user_dashboard, name="user_dashboard"),
    path("detection", user_views.detection, name="detection"),
    path("detection-result", user_views.detection_result, name="detection_result"),
    path("user-feedback", user_views.user_feedback, name="user_feedback"),
    path('user-logout',user_views.user_logout,name='user_logout'),
    path("profile", user_views.profile, name="profile"),
    # Admin
    path("admin-dashboard",admin_views.admin_dash, name="admin_dash"),
    path("all-users",admin_views.allusers, name="allusers"),
    path("pending-users",admin_views.pendingusers, name="pendingusers"),
    path("admin-feedback",admin_views.feedback, name="feedback"),
    path("graph-comparison",admin_views.graphcomparison, name="graphcomparison"),
    path("sentiment-analysis",admin_views.sentimentanalysis, name="sentimentanalysis"),
    path("sentiment-analysis-graph",admin_views.sentimentanalysisgraph, name="sentimentanalysisgraph"),
    path('delete-user/<int:user_id>/', admin_views.delete_user, name='delete_user'),
    path('accept-user/<int:id>', admin_views.accept_user, name = 'accept_user'),
    path('reject-user/<int:id>', admin_views.reject_user, name = 'reject'),
    path('change-status/<int:id>', admin_views.change_status, name = 'change_status'),
    path('admin-logout',admin_views.adminlogout,name='adminlogout'),
    path("naive_bayes", admin_views.naive_bayes, name="naive_bayes"),
    path("naive_bayes-result", admin_views.naive_bayes_result, name="naive_bayes_result"),
    path("multinomial-nb", admin_views.multinomial_nb, name="multinomial_nb"),
    path("multinomial-nb-result", admin_views.multinomial_nb_result, name="multinomial_nb_result"),
    path("decision-tree", admin_views.decision_tree, name="decision_tree"),
    path("decision-tree-result", admin_views.decision_tree_result, name="decision_tree_result"),
    path("random-forest", admin_views.random_forest, name="random_forest"),
    path("random-forest-result", admin_views.random_forest_result, name="random_forest_result"),
    path("logistic-regression", admin_views.logistic_regression, name="logistic_regression"),
    path("logistic-regression-result", admin_views.logistic_regression_result, name="logistic_regression_result"),
    path("admin-upload", admin_views.adminupload, name="admin_upload"),
    path("admin-view", admin_views.adminview, name="admin_view"),
    path("admin-view-result", admin_views.adminviewresult, name="admin_view_result"),
    path("delete-dataset/<int:id>", admin_views.delete_dataset, name="delete_dataset"),
    path("data-exploration", admin_views.dataexploration, name="data_exploration"),
] + static(settings.MEDIA_URL, document_root = settings.MEDIA_ROOT)


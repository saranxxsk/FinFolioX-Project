"""
URL configuration for finfolio_x project.
Phase 19: Includes the API routes under /api/
"""

from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include("api.urls")),  # Phase 19: REST API endpoints
]

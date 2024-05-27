from django import forms
from .models import Image


class UploadForm(forms.ModelForm):
    class Meta:
        model = Image
        fields = ['src']

    def __init__(self, *args, **kwargs):
        if "user" in kwargs:
            self.user = kwargs.pop('user')
        super().__init__(*args, **kwargs)

    def save(self, commit=True):
        Image.objects.create(user=self.user, src=self.cleaned_data['src'])

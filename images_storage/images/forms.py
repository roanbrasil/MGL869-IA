from django import forms
from django.forms import modelformset_factory
from .models import Image


class MultipleFileInput(forms.ClearableFileInput):
    allow_multiple_selected = True


class MultipleFileField(forms.FileField):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("widget", MultipleFileInput())
        super().__init__(*args, **kwargs)

    def clean(self, data, initial=None):
        single_file_clean = super().clean
        if isinstance(data, (list, tuple)):
            result = [single_file_clean(d, initial) for d in data]
        else:
            result = [single_file_clean(data, initial)]
        return result


class UploadForm(forms.Form):
    images = MultipleFileField()

    def __init__(self, *args, **kwargs):
        if "user" in kwargs:
            self.user = kwargs.pop("user")
        super().__init__(*args, **kwargs)

    def save(self, commit=True):
        for src in self.cleaned_data["images"]:
            Image.objects.create(user=self.user, src=src)


class DeleteForm(forms.ModelForm):
    class Meta:
        model = Image
        fields = "__all__"

    delete = forms.BooleanField(required=False)

    @property
    def image_src(self):
        return self.instance.src

    def save(self, commit=True):
        if bool(self.cleaned_data["delete"]):
            self.instance.delete()


DeleteFormSet = modelformset_factory(Image, form=DeleteForm, edit_only=True, extra=0)

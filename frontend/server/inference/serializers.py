from rest_framework import serializers


class ImageSerializer(serializers.Serializer):
    model = serializers.CharField(min_length=4, max_length=80, required=True)
    image = serializers.FileField(required=True)

    def update(self, instance, validated_data):
        raise NotImplementedError()

    def create(self, validated_data):
        return validated_data

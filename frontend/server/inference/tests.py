from unittest.mock import patch

from rest_framework.test import APISimpleTestCase


class PredictTestCase(APISimpleTestCase):
    def setUp(self):
        self.get_patcher = patch("requests.get")
        self.get_mock = self.get_patcher.start()

    def tearDown(self):
        self.get_patcher.stop()

    def test_predict(self):
        self.get_mock.return_value.json.return_value = {"models": [{"modelName": "some_model"}]}
        response = self.client.get("/inference/v1/predict/")
        self.assertEqual(response.status_code, 200)

# ADDRESS API SPEC

## CREATE ADDRESS

ENDPOINT: POST [Inference](http://127.0.0.1:5000/stream)

REQUEST HEADER:

- CONTENT-TYPE: multipart/form-data

REQUEST BODY:

```json
{
  "long": "example",
  "lat": "example",
  "image": "example",
}
```

RESPONSE BODY: (SUCCESS)

```json
{
    "message": "Prediction successful",
    "lat": "6.4025° S",
    "long": "106.7942° E",
    "prediction": "OtherCrack",
    "prediction_confidance": [
        [
            0.036640435457229614,
            0.02014610916376114,
            0.7645915746688843,
            0.04082467034459114,
            0.13779722154140472
        ]
    ],
    "images_base64": [
      "data:image/png;base64,<image_1_base64>",
      "data:image/png;base64,<image_2_base64>",
      "data:image/png;base64,<image_3_base64>"
    ]
}
```

RESPONSE BODY: (FAILED)

```json
{
  "message": "Error during prediction"
}
```

## GET ADDRESS

ENDPOINT: GET [Save Model](http://127.0.0.1:5000/save_model)

RESPONSE BODY: (SUCCESS)

```json
{
  "message": "Model Berhasil Di simpan"
}
```

RESPONSE BODY: (FAILED)

```json
{
  "errors": "Error there was a problem with the API"
}
```
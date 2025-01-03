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
    "lat": "Testing",
    "long": "Testing",
    "prediction_labeling": [
        "Patching",
        "Patching"
    ],
    "prediction_confidance": [
        {
            "Result-0": [
                [
                    0.0015741318929940462,
                    0.0002539046108722687,
                    0.0010026820236817002,
                    0.9931771755218506,
                    0.003992031794041395
                ]
            ]
        },
        {
            "Result-1": [
                [
                    8.6481828475371e-05,
                    9.349796528113075e-06,
                    7.156492938520387e-05,
                    0.9996932744979858,
                    0.00013937061885371804
                ]
            ]
        }
    ],
    "similarity_score": [
        {
            "Result-0": {
                "AlligatorCrack Rendah": 0.8045577626724885,
                "AlligatorCrack Sedang": 0.8256812905130726,
                "AlligatorCrack Tinggi": 0.9038883083572176
            }
        },
        {
            "Result-1": {
                "AlligatorCrack Rendah": 0.8427607528078315,
                "AlligatorCrack Sedang": 0.855156891948865,
                "AlligatorCrack Tinggi": 0.7720906824694109
            }
        }
    ],
    "bounding_boxes": [
      "data:image/png;base64,<image_1_base64>",
      "data:image/png;base64,<image_2_base64>",
      "data:image/png;base64,<image_3_base64>"
    ],
    "original_image": "data:image/png;base64,<image_1_base64>"
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
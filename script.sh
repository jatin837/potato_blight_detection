mkdir data
cd data
kaggle datasets download -d arjuntejaswi/plant-village


docker run -it -p 8050:8050 -v ~/dev/potato_blight_detection:/app tensorflow/serving --rest_api_port=8050 --model_config_file=/app/model.config

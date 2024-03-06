from cnn import logging
from cnn.components.data_ingestion import DataIngestionConfig, DataIngestion

STAGE_NAME = "Data Ingestion stage"

try:
    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")

    # Initialize configuration
    config = DataIngestionConfig()
    # Pass the config to DataIngestion
    data_ingestion = DataIngestion(config)
    data_ingestion.initiate_data_ingestion_and_save()

    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logging.exception(e)
    raise e

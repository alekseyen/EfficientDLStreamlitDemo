from minio import Minio
import os

class MINIO():

    def __init__(self):

        ip = os.environ['RETRAIN_IP']

        self.minioClient = Minio(f'{ip}:9000',
                    access_key='minio',
                    secret_key='minio123',
                    secure=False
                    )
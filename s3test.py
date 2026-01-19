from data.datasets.TUS_rec_s3 import TUSRecS3
from main_rec import load_dotenv
from s3torchconnector._s3client import S3Client, S3ClientConfig
from s3torchconnector import S3ReaderConstructor

load_dotenv('.env', override=False)

ds = TUSRecS3.__new__(TUSRecS3)
ds.bucket = 'tus-rec-24'
ds.reader_constructor = S3ReaderConstructor.default()
ds.client = S3Client(region='us-east-1', endpoint='http://172.16.240.77:9000',
                     s3client_config=S3ClientConfig(force_path_style=True))

frames, tforms = ds._load_frames_and_tforms(
    'train/frames/000/LH_Par_C_DtP.h5',
    'train/transfs/000/LH_Par_C_DtP.h5',
)
print(frames.shape, tforms.shape)

import sqlite3
import numpy as np
import pandas as pd
from psims.mzml.components import ParameterContainer, NullMap
from timsconvert.brukerapi.init_bruker_dll import *


MSMS_SPECTRUM_FUNCTOR = ctypes.CFUNCTYPE(None,
                                         ctypes.c_int64,
                                         ctypes.c_uint32,
                                         ctypes.POINTER(ctypes.c_double),
                                         ctypes.POINTER(ctypes.c_float))

MSMS_PROFILE_SPECTRUM_FUNCTOR = ctypes.CFUNCTYPE(None,
                                                 ctypes.c_int64,
                                                 ctypes.c_uint32,
                                                 ctypes.POINTER(ctypes.c_int32))


# modified from tsfdata.py
class tsf_data(object):
    def __init__(self, bruker_d_folder_name: str, bruker_dll, use_recalibrated_state=True):
        self.dll = bruker_dll
        self.handle = self.dll.tsf_open(bruker_d_folder_name.encode('utf-8'),
                                               1 if use_recalibrated_state else 0)
        if self.handle == 0:
            throw_last_tsf_error(self.dll)

        self.conn = sqlite3.connect(os.path.join(bruker_d_folder_name, 'analysis.tsf'))

        # arbitrary size, from Bruker tsfdata.py
        self.line_buffer_size = 1024
        self.profile_buffer_size = 1024

        self.meta_data = None
        self.frames = None
        self.maldiframeinfo = None
        self.framemsmsinfo = None
        self.source_file = bruker_d_folder_name

        self.get_global_metadata()

        self.get_frames_table()
        self.get_maldiframeinfo_table()
        self.get_framemsmsinfo_table()

    # from Bruker tsfdata.py
    def __del__(self):
        if hasattr(self, 'handle'):
            self.dll.tsf_close(self.handle)

    # from Bruker tsfdata.py
    def __call_conversion_func(self, frame_id, input_data, func):
        if type(input_data) is np.ndarray and input_data.dtype == np.float64:
            in_array = input_data
        else:
            in_array = np.array(input_data, dtype=np.float64)

        cnt = len(in_array)
        out = np.empty(shape=cnt, dtype=np.float64)
        success = func(self.handle,
                       frame_id,
                       in_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                       out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                       cnt)

        if success == 0:
            throw_last_tsf_error(self.dll)

        return out

    # from Bruker tsfdata.py
    def index_to_mz(self, frame_id, indices):
        return self.__call_conversion_func(frame_id, indices, self.dll.tsf_index_to_mz)

    # modified from Bruker tsfdata.py
    def read_line_spectrum(self, frame_id):
        while True:
            cnt = int(self.profile_buffer_size)
            index_buf = np.empty(shape=cnt, dtype=np.float64)
            intensity_buf = np.empty(shape=cnt, dtype=np.float32)

            required_len = self.dll.tsf_read_line_spectrum(self.handle,
                                                           frame_id,
                                                           index_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                                           intensity_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                                           self.profile_buffer_size)

            if required_len > self.profile_buffer_size:
                if required_len > 16777216:
                    raise RuntimeError('Maximum expected frame size exceeded.')
                self.profile_buffer_size = required_len
            else:
                break

        return (index_buf[0:required_len], intensity_buf[0:required_len])

    # modified from Bruker tsfdata.py
    # currently unsure of how to get m/z values/indices
    def read_profile_spectrum(self, frame_id):
        while True:
            cnt = int(self.profile_buffer_size)
            intensity_buf = np.empty(shape=cnt, dtype=np.uint32)

            required_len = self.dll.tsf_read_profile_spectrum(self.handle,
                                                              frame_id,
                                                              intensity_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                                                              self.profile_buffer_size)

            if required_len > self.profile_buffer_size:
                if required_len > 16777216:
                    raise RuntimeError('Maximum expected frame size exceeded.')
                self.profile_buffer_size = required_len
            else:
                break

        return intensity_buf[0:required_len]

    # Gets global metadata table as a dictionary.
    def get_global_metadata(self):
        metadata_query = 'SELECT * FROM GlobalMetadata'
        metadata_df = pd.read_sql_query(metadata_query, self.conn)
        metadata_dict = {}
        for index, row in metadata_df.iterrows():
            metadata_dict[row['Key']] = row['Value']
        self.meta_data = metadata_dict

    # Get Frames table from analysis.tsf SQL database.
    def get_frames_table(self):
        frames_query = 'SELECT * FROM Frames'
        self.frames = pd.read_sql_query(frames_query, self.conn)

    # Get MaldiFramesInfo table from analysis.tsf SQL database.
    def get_maldiframeinfo_table(self):
        maldiframeinfo_query = 'SELECT * FROM MaldiFrameInfo'
        self.maldiframeinfo = pd.read_sql_query(maldiframeinfo_query, self.conn)

    # Get FrameMsMsInfo table from analysis.tsf SQL database.
    def get_framemsmsinfo_table(self):
        framemsmsinfo_query = 'SELECT * FROM FrameMsMsInfo'
        self.framemsmsinfo = pd.read_sql_query(framemsmsinfo_query, self.conn)


if __name__ == '__main__':
    logger = logging.getLogger(__name__)

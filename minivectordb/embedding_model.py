from onnxruntime_extensions import get_library_path
import onnxruntime as ort
from os import cpu_count
import pkg_resources

class EmbeddingModel:
   
    def __init__(self, filepath = None):
        if filepath is None:
            filepath = pkg_resources.resource_filename('minivectordb', 'resources/embedding_model_quantized.onnx')
        self.model_path = filepath
        self.model = self.load_onnx_model()

    def load_onnx_model(self):

        _options = ort.SessionOptions()
        _options.inter_op_num_threads, _options.intra_op_num_threads = cpu_count(), cpu_count()
        _options.register_custom_ops_library(get_library_path())
        _providers = ["CPUExecutionProvider"]

        return ort.InferenceSession(
            path_or_bytes = self.model_path,
            sess_options=_options,
            providers=_providers
        )
    
    def extract_embeddings(self, text):
        return self.model.run(output_names=["outputs"], input_feed={"inputs": [text]})[0][0]

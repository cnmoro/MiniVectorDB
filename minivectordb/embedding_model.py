import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from onnxruntime_extensions import get_library_path
from torch import Tensor
import onnxruntime as ort
from os import cpu_count
import pkg_resources

class EmbeddingModel:
   
    def __init__(self, use_quantized_onnx_model = True, e5_model_size = 'small'):
        self.onnx_model_path = pkg_resources.resource_filename('minivectordb', 'resources/embedding_model_quantized.onnx')
        self.use_quantized_onnx_model = use_quantized_onnx_model
        self.e5_model_size = e5_model_size

        if self.use_quantized_onnx_model:
            self.load_onnx_model()
        else:
            self.load_e5_multi_model()

    def load_onnx_model(self):
        _options = ort.SessionOptions()
        _options.inter_op_num_threads, _options.intra_op_num_threads = cpu_count(), cpu_count()
        _options.register_custom_ops_library(get_library_path())
        _providers = ["CPUExecutionProvider"]

        self.model = ort.InferenceSession(
            path_or_bytes = self.onnx_model_path,
            sess_options=_options,
            providers=_providers
        )

    def average_pool(self, last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def load_e5_multi_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(f'intfloat/multilingual-e5-{self.e5_model_size}')
        self.model = AutoModel.from_pretrained(f'intfloat/multilingual-e5-{self.e5_model_size}')
    
    def extract_embeddings_e5_multi(self, text):
        # Tokenize the input texts
        batch_dict = self.tokenizer([f'passage {text}'], max_length=512, padding=True, truncation=True, return_tensors='pt')

        outputs = self.model(**batch_dict)
        embeddings = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        # normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.tolist()[0]

    def extract_embeddings_quant_onnx(self, text):
        return self.model.run(output_names=["outputs"], input_feed={"inputs": [text]})[0][0]
    
    def extract_embeddings(self, text):
        if self.use_quantized_onnx_model:
            return self.extract_embeddings_quant_onnx(text)
        else:
            return self.extract_embeddings_e5_multi(text)
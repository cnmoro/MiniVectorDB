import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from FlagEmbedding import BGEM3FlagModel
from onnxruntime_extensions import get_library_path
from torch import Tensor
import onnxruntime as ort
from os import cpu_count
import pkg_resources
from enum import Enum

class AlternativeModel(str, Enum):
    small = "small"
    large = "large"
    bgem3 = "bgem3"

class EmbeddingModel:
   
    def __init__(self, use_quantized_onnx_model = True, alternative_model: AlternativeModel = AlternativeModel.bgem3, onnx_model_cpu_core_count=None, **kwargs):
        self.onnx_model_path = pkg_resources.resource_filename('minivectordb', 'resources/embedding_model_quantized.onnx')
        self.use_quantized_onnx_model = use_quantized_onnx_model
        self.onnx_model_cpu_core_count = onnx_model_cpu_core_count

        assert isinstance(self.onnx_model_cpu_core_count, int) or self.onnx_model_cpu_core_count is None
        
        # Check if "e5_model_size" is in kwargs
        # We changed this parameter, but we want to keep the old one for compatibility
        if 'e5_model_size' in kwargs:
            self.alternative_model = AlternativeModel(kwargs['e5_model_size'])
        else:
            self.alternative_model = alternative_model

        if self.use_quantized_onnx_model:
            self.load_onnx_model()
        else:
            self.load_alternative_model()

    def load_onnx_model(self):
        cpu_core_count = cpu_count() if self.onnx_model_cpu_core_count is None else self.onnx_model_cpu_core_count
        _options = ort.SessionOptions()
        _options.inter_op_num_threads, _options.intra_op_num_threads = cpu_core_count, cpu_core_count
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

    def load_alternative_model(self):
        if self.alternative_model == AlternativeModel.small or self.alternative_model == AlternativeModel.large:
            self.tokenizer = AutoTokenizer.from_pretrained(f'intfloat/multilingual-e5-{self.alternative_model.value}')
            self.model = AutoModel.from_pretrained(f'intfloat/multilingual-e5-{self.alternative_model.value}')
        elif self.alternative_model == AlternativeModel.bgem3:
            self.model = BGEM3FlagModel('BAAI/bge-m3')
    
    def extract_embeddings_e5_multi(self, text):
        # Tokenize the input texts
        batch_dict = self.tokenizer([f'passage {text}'], max_length=512, padding=True, truncation=True, return_tensors='pt')

        outputs = self.model(**batch_dict)
        embeddings = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        # normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.tolist()[0]

    def extract_embeddings_bgem3(self, text):
        embeddings = self.model.encode(
            [text], 
            batch_size=1, 
            max_length=512,
            )['dense_vecs']
        return embeddings[0].tolist()

    def extract_embeddings_quant_onnx(self, text):
        return self.model.run(output_names=["outputs"], input_feed={"inputs": [text]})[0][0]
    
    def extract_embeddings(self, text):
        if self.use_quantized_onnx_model:
            return self.extract_embeddings_quant_onnx(text)
        else:
            if self.alternative_model == AlternativeModel.small or self.alternative_model == AlternativeModel.large:
                return self.extract_embeddings_e5_multi(text)
            else:
                return self.extract_embeddings_bgem3(text)

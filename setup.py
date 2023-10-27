from setuptools import setup, find_packages

setup(
    name='minivectordb',
    version='1.0.2',
    author='Carlo Moro',
    author_email='cnmoro@gmail.com',
    description="This is a Python project aimed at extracting embeddings from textual data and performing semantic search. It's a simple yet powerful system combining a small quantized ONNX model with FAISS indexing for fast similarity search. As the model is small and also running in ONNX runtime with quantization, we get lightning fast speed.",
    packages=find_packages(),
    package_data={
        'minivectordb': ['resources/embedding_model_quantized.onnx']
    },
    include_package_data=True,
    install_requires=[
        "numpy==1.26.1",
        "onnx==1.15.0",
        "onnxruntime==1.16.1",
        "onnxruntime-extensions==0.9.0",
        "faiss-cpu==1.7.4",
        "pytest==7.4.3",
        "pytest-cov==4.1.0"
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
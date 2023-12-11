from setuptools import setup, find_packages

setup(
    name='minivectordb',
    version='1.1.3',
    author='Carlo Moro',
    author_email='cnmoro@gmail.com',
    description="This is a Python project aimed at extracting embeddings from textual data and performing semantic search.",
    packages=find_packages(),
    package_data={
        'minivectordb': ['resources/embedding_model_quantized.onnx']
    },
    include_package_data=True,
    install_requires=[
        "numpy",
        "onnx",
        "onnxruntime",
        "onnxruntime-extensions",
        "transformers",
        "faiss-cpu",
        "torch",
        "pytest",
        "pytest-cov",
        "rank-bm25",
        "thefuzz[speedup]"
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
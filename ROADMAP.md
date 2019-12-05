# TransmogrifAI Roadmap

## Short Term

- Implement Python interface for loading and evaluating TransmogrifAI models - [#393](https://github.com/salesforce/TransmogrifAI/issues/393)
- Extend Python interface to allow defining workflows & train models with existing TransmogrifAI readers, stages etc. - [#393](https://github.com/salesforce/TransmogrifAI/issues/393)
- Standatize versioning of TransmogrifAI models format and implement verification for safe execution - [#397](https://github.com/salesforce/TransmogrifAI/issues/397)
- Implement deep learning support, e.g. an ability to load an TensorFlow models and score them - [#288](https://github.com/salesforce/TransmogrifAI/issues/248), [#355](https://github.com/salesforce/TransmogrifAI/pull/355)
- Investigate integration with [JohnSnowNLP](https://github.com/JohnSnowLabs/spark-nlp) library, since it has some many fancy operations that we could use



## Long Term

- Extend Python interface to allow defining custom readers & stages with Python code snippets - [#393](https://github.com/salesforce/TransmogrifAI/issues/393)
- Allow exporting TransmogrifAI models in MLeap or other popular format, e.g. ONNX

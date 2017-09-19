# scala-tensorflow-predict

This is meant as a blueprint for someone to use the Tensorflow Java API to predict a Tensorflow model within a Scala environment. The included class definitely needs much more work for it to be production-ready but it's an end-to-end guideline that worked for me (in essence a summary of learnings from tensorflow docs, stackoverflow answers, and my own debugging). 

Besides the Scala predictor file in this repo, here is what is necessary:
1. Google Protobuf definitions: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/example/feature.proto and 
```
message PFeaturesForPredict {
  Features features = 1;
};
```
2. Save your model in python:
```python
  feature_spec = [tf.feature_column.numeric_column("feature_name1"), tf.feature_column.numeric_column("feature_name2")]
  serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
    feature_spec
  )
  saved_model_dir = classifier.export_savedmodel("/tmp/exported_models", serving_input_receiver_fn)
```
3. Check out the schema of your saved model for input/output names by executing the tensorflow command line utility:
`saved_model_cli show --dir /tmp/saved_model_dir --tag_set serve --signature_def serving_default`


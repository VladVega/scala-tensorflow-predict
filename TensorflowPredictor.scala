package tensor_java_predictor

import org.tensorflow.{DataType, Graph, Output, SavedModelBundle, Session, Tensor}
import presence.protobuf.Feature.Kind
import presence.protobuf.{Feature, Features, FloatList}
import presence.protobuf.presenceactor.PFeaturesForPredict

case class TensorflowPredictor() {
  val inputName = "input_example_tensor:0"
  val outputName = "linear/head/predictions/probabilities:0"

  val pathToModel = getModelsPath()
  val session = SavedModelBundle.load(pathToModel, "serve").session()
  val reshaper = TensorReshaper()

  def predict(featuresRaw: Seq[(String, Double)]): Double = {
    val proto = createProto(featuresRaw)
    val stringTensor = Tensor.create(proto.toByteArray)
    val inputTensor = reshaper.vector(stringTensor)

    val output = session.runner()
      .feed(inputName, inputTensor)
      .fetch(outputName)
      .run()

    val probabilityArr = output.get(0).copyTo(Array.ofDim[Float](1,2))
    val probability = probabilityArr(0)(1) //probability that label=1
    probability
  }

  def getModelsPath(): String = {
    //selects the latest model in the dir
    new java.io.File("path_to_models_dir")).filter(_.exists()).head.listFiles.last.toString
  }

  def createProto(featuresRaw: Seq[(String, Double)]): PFeaturesForPredict = {
    val featureMap = featuresRaw.foldLeft(Map[String, Feature]()) { (resMap, feature) =>
      val fname = feature._1
      val value: Double = feature._2

      resMap + (fname -> Feature(Kind.FloatList(FloatList(Seq(value.toFloat)))))
    }

    PFeaturesForPredict(Option(Features(featureMap)))
  }

}

//from https://stackoverflow.com/questions/45746742/how-can-i-create-a-tensor-from-an-example-object-in-java
//https://github.com/asimshankar/java-tensorflow/tree/master/string_vector_workaround
//Converted the above java code to scala.
case class TensorReshaper() extends AutoCloseable {
  val graph = new Graph()
  val session = new Session(graph)
  val in = graph.opBuilder("Placeholder", "in").setAttr("dtype", DataType.STRING).build.output(0)
  val shape: Tensor = Tensor.create(Array[Int](1))
  val vectorShape: Output = graph.opBuilder("Const", "vector_shape").setAttr("dtype", shape.dataType).setAttr("value", shape).build.output(0)
  val out = graph.opBuilder("Reshape", "out").addInput(in).addInput(vectorShape).build.output(0)

  def close {
    session.close
    graph.close
  }

  def vector(input: Tensor): Tensor = {
    return session
      .runner
      .feed(in, input)
      .fetch(out).run.get(0)
  }

}
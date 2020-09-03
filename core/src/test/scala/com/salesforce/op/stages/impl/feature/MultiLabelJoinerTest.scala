package com.salesforce.op.stages.impl.feature

import com.salesforce.op.features.types._
import com.salesforce.op.test.{OpTransformerSpec, TestFeatureBuilder}
import org.apache.spark.ml.linalg.Vectors
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

import scala.reflect.ClassTag

@RunWith(classOf[JUnitRunner])
class MultiLabelJoinerTest extends MultiLabelJoinerBaseTest[MultiLabelJoiner] {
  val transformer = new MultiLabelJoiner().setInput(classIndexFeature, probVecFeature)

  val expectedResult = Seq(
    classes.zip(Array(40.0, 30.0, 20.0, 0.0)).toMap.toRealMap,
    classes.zip(Array(20.0, 40.0, 30.0, 0.0)).toMap.toRealMap,
    classes.zip(Array(30.0, 20.0, 40.0, 0.0)).toMap.toRealMap
  )
}

abstract class MultiLabelJoinerBaseTest[T <: MultiLabelJoiner : ClassTag] extends OpTransformerSpec[RealMap, T] {
  // Input Dataset and features
  val (inputDF, idFeature, classFeature, probVecFeature) = TestFeatureBuilder("ID", "class", "prob",
    Seq[(Integral, Text, OPVector)](
      (Integral(1001), Text("Low"), OPVector(Vectors.dense(Array(40.0, 30.0, 20.0, 0.0)))),
      (Integral(1002), Text("Medium"), OPVector(Vectors.dense(Array(20.0, 40.0, 30.0, 0.0)))),
      (Integral(1003), Text("High"), OPVector(Vectors.dense(Array(30.0, 20.0, 40.0, 0.0))))
    )
  )
  val classIndexFeature = classFeature.indexed(unseenName = OpStringIndexerNoFilter.UnseenNameDefault)

  // String indexer stage estimator.
  val indexStage = classIndexFeature.originStage.asInstanceOf[OpStringIndexerNoFilter[_]].fit(inputDF)
  val inputData = indexStage.transform(inputDF)

  // Apart from classes in the data - Low, High, Medium, there is an additional class - UnseenLabel for unseen classes.
  val classes = indexStage.getMetadata().getMetadata("ml_attr").getStringArray("vals")
}

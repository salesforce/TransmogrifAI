package com.salesforce.op.stages.impl.feature

import com.salesforce.op.features.types._
import com.salesforce.op.test.TestOpVectorColumnType.IndVal
import com.salesforce.op.test.{TestFeatureBuilder, TestOpVectorMetadataBuilder, TestSparkContext}
import com.salesforce.op.utils.spark.OpVectorMetadata
import com.salesforce.op.utils.spark.RichDataset._
import org.apache.spark.ml.linalg.Vectors
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class TextLenTransformerTest extends FlatSpec with TestSparkContext with AttributeAsserts {
  val (ds, f1, f2) = TestFeatureBuilder(
    Seq[(TextList, TextList)](
      (TextList(Seq("A giraffe drinks by the watering hole")),
        TextList(Seq("A giraffe drinks by the watering hole"))),
      (TextList(Seq("A giraffe drinks by the watering hole")), TextList(Seq("Cheese"))),
      (TextList(Seq("Cheese", "cake")), TextList(Seq("A giraffe drinks by the watering hole"))),
      (TextList(Seq("Cheese")), TextList(Seq("Cheese"))),
      (TextList.empty, TextList(Seq("A giraffe drinks by the watering hole"))),
      (TextList.empty, TextList(Seq("Cheese", "tart"))),
      (TextList(Seq("A giraffe drinks by the watering hole")), TextList.empty),
      (TextList(Seq("Cheese")), TextList.empty),
      (TextList.empty, TextList.empty)
    )
  )

  Spec[TextLenTransformer[_]] should "take an array of features as input and return a single vector feature" in {
    val vectorizer = new TextLenTransformer().setInput(f1, f2)
    val vector = vectorizer.getOutput()

    vector.name shouldBe vectorizer.getOutputFeatureName
    vector.typeName shouldBe FeatureType.typeName[OPVector]
    vector.isResponse shouldBe false
  }

  it should "transform the data correctly" in {
    val vectorizer = new TextLenTransformer().setInput(f1, f2)
    val transformed = vectorizer.transform(ds)
    val vector = vectorizer.getOutput()

    val expected = Array(
      Array(37.0, 37.0),
      Array(37.0, 6.0),
      Array(10.0, 37.0),
      Array(6.0, 6.0),
      Array(0.0, 37.0),
      Array(0.0, 10.0),
      Array(37.0, 0.0),
      Array(6.0, 0.0),
      Array(0.0, 0.0)
    ).map(Vectors.dense(_).toOPVector)
    val result = transformed.collect(vector)
    result shouldBe expected

    val vectorMetadata = vectorizer.getMetadata()
    OpVectorMetadata(vectorizer.getOutputFeatureName, vectorMetadata) shouldEqual TestOpVectorMetadataBuilder(
      vectorizer,
      f1 -> List(IndVal(Some(TransmogrifierDefaults.TextLenString))),
      f2 -> List(IndVal(Some(TransmogrifierDefaults.TextLenString)))
    )
  }
}

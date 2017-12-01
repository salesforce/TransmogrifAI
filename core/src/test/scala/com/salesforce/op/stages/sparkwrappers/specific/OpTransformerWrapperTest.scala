/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.sparkwrappers.specific

import com.salesforce.op.features.types._
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import com.salesforce.op.utils.spark.RichDataset._
import org.apache.spark.ml.feature.{Normalizer, StopWordsRemover}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{FlatSpec, Matchers}

@RunWith(classOf[JUnitRunner])
class OpTransformerWrapperTest extends FlatSpec with TestSparkContext {

  val (testData, featureVector) = TestFeatureBuilder(
    Seq[MultiPickList](
      Set("I", "saw", "the", "red", "balloon").toMultiPickList,
      Set("Mary", "had", "a", "little", "lamb").toMultiPickList
    )
  )

  val (testDataNorm, _, _) = TestFeatureBuilder("label", "features",
    Seq[(Real, OPVector)](
      0.0.toReal -> Vectors.dense(1.0, 0.5, -1.0).toOPVector,
      1.0.toReal -> Vectors.dense(2.0, 1.0, 1.0).toOPVector,
      2.0.toReal -> Vectors.dense(4.0, 10.0, 2.0).toOPVector
    )
  )
  val (targetDataNorm, targetLabelNorm, featureVectorNorm) = TestFeatureBuilder("label", "features",
    Seq[(Real, OPVector)](
      0.0.toReal -> Vectors.dense(0.4, 0.2, -0.4).toOPVector,
      1.0.toReal -> Vectors.dense(0.5, 0.25, 0.25).toOPVector,
      2.0.toReal -> Vectors.dense(0.25, 0.625, 0.125).toOPVector
    )
  )

  Spec[OpTransformerWrapper[_, _, _]] should "remove stop words with caseSensitivity=true" in {
    val remover = new StopWordsRemover().setCaseSensitive(true)
    val swFilter =
      new OpTransformerWrapper[MultiPickList, MultiPickList, StopWordsRemover](remover).setInput(featureVector)
    val output = swFilter.transform(testData)
    output.show(false)

    output.collect(swFilter.getOutput()) shouldBe Array(
      Seq("I", "saw", "red", "balloon").toMultiPickList,
      Seq("Mary", "little", "lamb").toMultiPickList
    )
  }

  it should "should properly normalize each feature vector instance with non-default norm of 1" in {
    val baseNormalizer = new Normalizer().setP(1.0)
    val normalizer =
      new OpTransformerWrapper[OPVector, OPVector, Normalizer](baseNormalizer).setInput(featureVectorNorm)
    val output = normalizer.transform(testDataNorm)
    output.show(false)

    val sumSqDist = validateDataframeDoubleColumn(output, normalizer.getOutput().name, targetDataNorm, "features")
    assert(sumSqDist <= 1E-6, "==> the sum of squared distances between actual and expected should be below tolerance.")
  }

  def validateDataframeDoubleColumn(
    normalizedFeatureDF: DataFrame, normalizedFeatureName: String, targetFeatureDF: DataFrame, targetColumnName: String
  ): Double = {
    val sqDistUdf = udf { (leftColVec: Vector, rightColVec: Vector) => Vectors.sqdist(leftColVec, rightColVec) }

    val targetColRename = "targetFeatures"
    val renamedTargedDF = targetFeatureDF.withColumnRenamed(targetColumnName, targetColRename)
    val joinedDF = normalizedFeatureDF.join(renamedTargedDF, Seq("label"))

    joinedDF.show(false)

    // compute sum of squared distances between expected and actual
    val finalDF = joinedDF.withColumn("sqDist", sqDistUdf(joinedDF(normalizedFeatureName), joinedDF(targetColRename)))
    val sumSqDist: Double = finalDF.agg(sum(finalDF("sqDist"))).first().getDouble(0)
    sumSqDist
  }
}

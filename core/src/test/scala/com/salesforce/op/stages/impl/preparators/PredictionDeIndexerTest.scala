/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.preparators


import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.unary.UnaryLambdaTransformer
import com.salesforce.op.stages.impl.feature.OpStringIndexerNoFilter
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import org.scalatest.FlatSpec

class PredictionDeIndexerTest extends FlatSpec with TestSparkContext {

  val data = Seq(("a", 0.0), ("b", 1.0), ("c", 2.0)).map { case (txt, num) => (txt.toText, num.toRealNN) }
  val (ds, txtF, numF) = TestFeatureBuilder(data)

  val response = txtF.indexed()
  val indexedData = response.originStage.asInstanceOf[OpStringIndexerNoFilter[_]].fit(ds).transform(ds)

  val permutation = new UnaryLambdaTransformer[RealNN, RealNN](
    operationName = "modulo",
    transformFn = v => ((v.value.get + 1).toInt % 3).toRealNN
  ).setInput(response)
  val pred = permutation.getOutput()
  val permutedData = permutation.transform(indexedData)

  val expected = Array("b", "c", "a").map(_.toText)

  Spec[PredictionDeIndexer] should "deindexed the feature correctly" in {
    val predDeIndexer = new PredictionDeIndexer().setInput(response, pred)
    val deIndexed = predDeIndexer.getOutput()

    val results = predDeIndexer.fit(permutedData).transform(permutedData).collect(deIndexed)
    results shouldBe expected
  }


  it should "throw a nice error when there is no metadata" in {
    val predDeIndexer = new PredictionDeIndexer().setInput(numF, pred)
    the[Error] thrownBy {
      predDeIndexer.fit(permutedData).transform(permutedData)
    } should have message
      s"The feature ${numF.name} does not contain any label/index mapping in its metadata"
  }
}

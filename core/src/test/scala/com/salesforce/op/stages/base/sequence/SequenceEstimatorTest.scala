/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of Salesforce.com nor the names of its contributors may
 * be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op.stages.base.sequence

import com.salesforce.op.UID
import com.salesforce.op.features.Feature
import com.salesforce.op.features.types._
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.utils.spark.SequenceAggregators
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.Dataset
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class SequenceEstimatorTest extends FlatSpec with TestSparkContext {

  val data = Seq[(DateList, DateList, DateList)](
    (new DateList(1476726419000L, 1476726019000L),
      new DateList(1476726919000L),
      new DateList(1476726519000L)),
    (new DateList(1476725419000L, 1476726019000L),
      new DateList(1476726319000L, 1476726919000L),
      new DateList(1476726419000L)),
    (new DateList(1476727419000L),
      new DateList(1476728919000L),
      new DateList(1476726619000L, 1476726949000L))
  )
  val (ds, clicks, opens, purchases) = TestFeatureBuilder("clicks", "opens", "purchases", data)

  val testEstimator: SequenceEstimator[DateList, OPVector] = new FractionOfResponsesEstimator()

  Spec[SequenceEstimator[_, _]] should "throw an error if you try to get the output without setting the inputs" in {
    intercept[java.util.NoSuchElementException](testEstimator.getOutput())
  }

  it should "return a single output feature of the correct type" in {
    val outputFeatures = testEstimator.setInput(clicks, opens, purchases).getOutput()
    outputFeatures shouldBe new Feature[OPVector](
      name = testEstimator.getOutputFeatureName,
      originStage = testEstimator,
      isResponse = false,
      parents = Array(clicks, opens, purchases)
    )
  }

  it should "return a SequenceModel with the estimator as the parent and the correct function" in {
    val testModel = testEstimator.setInput(clicks, opens, purchases).fit(ds)
    testModel.parent shouldBe testEstimator
    testModel.transformFn(
      Seq(new DateList(1476726419000L), new DateList(1476726419000L), new DateList(1476726419000L))
    ) shouldEqual Vectors.dense(0.2, 0.25, 0.25).toOPVector
  }

  it should "create a SequenceModel that uses the specified transform function when fit" in {
    val testModel = testEstimator.setInput(clicks, opens, purchases).fit(ds)
    val testDataTransformed = testModel.setInput(clicks, opens, purchases).transform(ds)
    val transformedValues = testDataTransformed.collect(clicks, opens, purchases, testModel.getOutput())

    // This is string because of vector type being private to spark ml
    testDataTransformed.schema.fieldNames.toSet shouldEqual
      Set(clicks.name, opens.name, purchases.name, testEstimator.getOutputFeatureName)

    val fractions = Array(
      Vectors.dense(0.4, 0.25, 0.25).toOPVector,
      Vectors.dense(0.4, 0.5, 0.25).toOPVector,
      Vectors.dense(0.2, 0.25, 0.5).toOPVector
    )
    val expected = data.zip(fractions) .map { case ((d1, d2, d3), f) => (d1, d2, d3, f)}

    transformedValues shouldBe expected
  }

  it should "copy itself and the model successfully" in {
    val est = new FractionOfResponsesEstimator()
    val mod = new FractionOfResponsesModel(Seq.empty, est.operationName, est.uid)

    est.copy(new ParamMap()).uid shouldBe est.uid
    mod.copy(new ParamMap()).uid shouldBe mod.uid
  }
}

class FractionOfResponsesEstimator(uid: String = UID[FractionOfResponsesEstimator])
  extends SequenceEstimator[DateList, OPVector](operationName = "fractionOfResponses", uid = uid) {
  def fitFn(dataset: Dataset[Seq[Seq[Long]]]): SequenceModel[DateList, OPVector] = {
    import dataset.sparkSession.implicits._
    val sizes = dataset.map(_.map(_.size))
    val size = getInputFeatures().length
    val counts = sizes.select(SequenceAggregators.SumNumSeq[Int](size = size).toColumn).first()
    new FractionOfResponsesModel(counts = counts, operationName = operationName, uid = uid)
  }
}

final class FractionOfResponsesModel private[op]
(
  val counts: Seq[Int],
  operationName: String,
  uid: String
) extends SequenceModel[DateList, OPVector](operationName = operationName, uid = uid) {
  def transformFn: (Seq[DateList]) => OPVector = row => {
    val fractions = row.zip(counts).map { case (feature, count) => feature.value.size.toDouble / count }
    Vectors.dense(fractions.toArray).toOPVector
  }
}


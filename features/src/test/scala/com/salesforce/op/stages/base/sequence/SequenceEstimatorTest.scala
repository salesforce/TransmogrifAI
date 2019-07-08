/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op.stages.base.sequence

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.test.{OpEstimatorSpec, TestFeatureBuilder}
import com.salesforce.op.utils.spark.SequenceAggregators
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.Dataset
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class SequenceEstimatorTest
  extends OpEstimatorSpec[OPVector, SequenceModel[DateList, OPVector], SequenceEstimator[DateList, OPVector]] {

  val sample = Seq[(DateList, DateList, DateList)](
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
  val (inputData, clicks, opens, purchases) = TestFeatureBuilder("clicks", "opens", "purchases", sample)

  val estimator = new FractionOfResponsesEstimator().setInput(clicks, opens, purchases)

  val expectedResult = Seq(
    Vectors.dense(0.4, 0.25, 0.25).toOPVector,
    Vectors.dense(0.4, 0.5, 0.25).toOPVector,
    Vectors.dense(0.2, 0.25, 0.5).toOPVector
  )
}


class FractionOfResponsesEstimator(uid: String = UID[FractionOfResponsesEstimator])
  extends SequenceEstimator[DateList, OPVector](operationName = "fractionOfResponses", uid = uid) {
  def fitFn(dataset: Dataset[Seq[Seq[Long]]]): SequenceModel[DateList, OPVector] = {
    import dataset.sparkSession.implicits._
    val sizes = dataset.map(_.map(_.size))
    val size = getInputFeatures().length
    val counts = sizes.select(SequenceAggregators.SumNumSeq[Int](size = size).toColumn).first().toList
    new FractionOfResponsesModel(counts = counts, operationName = operationName, uid = uid)
  }
}

final class FractionOfResponsesModel private[op]
(
  val counts: List[Int],
  operationName: String,
  uid: String
) extends SequenceModel[DateList, OPVector](operationName = operationName, uid = uid) {
  def transformFn: Seq[DateList] => OPVector = row => {
    val fractions = row.zip(counts).map { case (feature, count) => feature.value.size.toDouble / count }
    Vectors.dense(fractions.toArray).toOPVector
  }
}


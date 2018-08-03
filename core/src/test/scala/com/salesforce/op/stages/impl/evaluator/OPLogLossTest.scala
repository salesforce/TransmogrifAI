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

package com.salesforce.op.stages.impl.evaluator

import com.salesforce.op.features.types._
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import org.apache.spark.ml.linalg.Vectors
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class OPLogLossTest extends FlatSpec with TestSparkContext {

  val (ds, rawLabel, raw, prob, pred) = TestFeatureBuilder[RealNN, OPVector, OPVector, RealNN](
    Seq(
      (1.0, Vectors.dense(8, 1, 1), Vectors.dense(0.8, 0.1, 0.1), 0.0),
      (0.0, Vectors.dense(1.0, 0.0, 0.0), Vectors.dense(1.0, 0.0, 0.0), 0.0),
      (0.0, Vectors.dense(1.0, 0.8, 0.2), Vectors.dense(0.5, 0.4, 0.1), 0.0),
      (1.0, Vectors.dense(10.0, 80.0, 10.0), Vectors.dense(0.1, 0.8, 0.1), 1.0),
      (2.0, Vectors.dense(0.0, 0.0, 14.0), Vectors.dense(0.0, 0.0, 1.0), 2.0),
      (2.0, Vectors.dense(0.0, 0.0, 13.0), Vectors.dense(0.0, 0.0, 1.0), 2.0),
      (1.0, Vectors.dense(0.1, 0.4, 0.5), Vectors.dense(0.1, 0.4, 0.5), 2.0),
      (0.0, Vectors.dense(0.1, 0.6, 0.3), Vectors.dense(0.1, 0.6, 0.3), 1.0),
      (1.0, Vectors.dense(1.0, 0.8, 0.2), Vectors.dense(0.5, 0.4, 0.1), 0.0),
      (2.0, Vectors.dense(1.0, 0.8, 0.2), Vectors.dense(0.5, 0.4, 0.1), 0.0)
    ).map(v => (v._1.toRealNN, v._2.toOPVector, v._3.toOPVector, v._4.toRealNN))
  )

  val label = rawLabel.copy(isResponse = true)
  val expected: Double = -math.log(0.1 * 0.5 * 0.8 * 0.4 * 0.1 * 0.4 * 0.1) / 10.0

  val (dsEmpty, rawLabelEmpty, rawEmpty, probEmpty, predEmpty) = TestFeatureBuilder[RealNN, OPVector, OPVector, RealNN](
    Seq()
  )

  val labelEmpty = rawLabel.copy(isResponse = true)


  val logLoss = LogLoss.mulitLogLoss

  Spec(LogLoss.getClass) should "compute logarithmic loss metric" in {
    val metric = logLoss.setLabelCol(label).setPredictionCol(pred).setRawPredictionCol(raw).setProbabilityCol(prob)
      .evaluate(ds)
    metric shouldBe expected

  }

  it should "throw an error when the dataset is empty" in {

    the[IllegalArgumentException] thrownBy {
      logLoss.setLabelCol(labelEmpty).setPredictionCol(predEmpty).setRawPredictionCol(rawEmpty)
        .setProbabilityCol(probEmpty).evaluate(dsEmpty)
    } should have message
      s"Metric ${logLoss.name} failed on empty dataset with (${labelEmpty.name}, ${rawEmpty.name}, ${probEmpty.name}," +
        s" ${predEmpty.name})"
  }
}

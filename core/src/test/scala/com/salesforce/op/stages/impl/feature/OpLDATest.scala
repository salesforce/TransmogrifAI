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

package com.salesforce.op.stages.impl.feature

import com.salesforce.op._
import com.salesforce.op.features.types._
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import com.salesforce.op.utils.spark.RichDataset._
import org.apache.spark.ml.clustering.LDA
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{Assertions, FlatSpec, Matchers}


@RunWith(classOf[JUnitRunner])
class OpLDATest extends FlatSpec with TestSparkContext {

  val inputData = Seq(
    (0.0, Vectors.sparse(11, Array(0, 1, 2, 4, 5, 6, 7, 10), Array(1.0, 2.0, 6.0, 2.0, 3.0, 1.0, 1.0, 3.0))),
    (1.0, Vectors.sparse(11, Array(0, 1, 3, 4, 7, 10), Array(1.0, 3.0, 1.0, 3.0, 2.0, 1.0))),
    (2.0, Vectors.sparse(11, Array(0, 1, 2, 5, 6, 8, 9), Array(1.0, 4.0, 1.0, 4.0, 9.0, 1.0, 2.0))),
    (3.0, Vectors.sparse(11, Array(0, 1, 3, 6, 8, 9, 10), Array(2.0, 1.0, 3.0, 5.0, 2.0, 3.0, 9.0))),
    (4.0, Vectors.sparse(11, Array(0, 1, 2, 3, 4, 6, 9, 10), Array(3.0, 1.0, 1.0, 9.0, 3.0, 2.0, 1.0, 3.0))),
    (5.0, Vectors.sparse(11, Array(0, 1, 3, 4, 5, 6, 7, 8, 9), Array(4.0, 2.0, 3.0, 4.0, 5.0, 1.0, 1.0, 1.0, 4.0))),
    (6.0, Vectors.sparse(11, Array(0, 1, 3, 6, 8, 9, 10), Array(2.0, 1.0, 3.0, 5.0, 2.0, 2.0, 9.0))),
    (7.0, Vectors.sparse(11, Array(0, 1, 2, 3, 4, 5, 6, 9, 10), Array(1.0, 1.0, 1.0, 9.0, 2.0, 1.0, 2.0, 1.0, 3.0))),
    (8.0, Vectors.sparse(11, Array(0, 1, 3, 4, 5, 6, 7), Array(4.0, 4.0, 3.0, 4.0, 2.0, 1.0, 3.0))),
    (9.0, Vectors.sparse(11, Array(0, 1, 2, 4, 6, 8, 9, 10), Array(2.0, 8.0, 2.0, 3.0, 2.0, 2.0, 7.0, 2.0))),
    (10.0, Vectors.sparse(11, Array(0, 1, 2, 3, 5, 6, 9, 10), Array(1.0, 1.0, 1.0, 9.0, 2.0, 2.0, 3.0, 3.0))),
    (11.0, Vectors.sparse(11, Array(0, 1, 4, 5, 6, 7, 9), Array(4.0, 1.0, 4.0, 5.0, 1.0, 3.0, 1.0)))
  ).map(v => v._1.toReal -> v._2.toOPVector)

  lazy val (ds, f1, f2) = TestFeatureBuilder(inputData)

  lazy val inputDS = ds.persist()

  val seed = 1234567890L
  val k = 3
  val maxIter = 100

  lazy val expected = new LDA()
    .setFeaturesCol(f2.name)
    .setK(k)
    .setSeed(seed)
    .fit(inputDS)
    .transform(inputDS)
    .select("topicDistribution")
    .collect()
    .toSeq
    .map(_.getAs[Vector](0))

  Spec[OpLDA] should "convert document term vectors into topic vectors" in {
    val f2Vec = new OpLDA().setInput(f2).setK(k).setSeed(seed).setMaxIter(maxIter)
    val testTransformedData = f2Vec.fit(inputDS).transform(inputDS)
    val output = f2Vec.getOutput()
    val estimate = testTransformedData.collect(output)
    val mse = computeMeanSqError(estimate, expected)
    val expectedMse = 0.5
    withClue(s"Computed mse $mse (expected $expectedMse)") {
      mse should be < expectedMse
    }
  }

  it should "convert document term vectors into topic vectors (shortcut version)" in {
    val output = f2.lda(k = k, seed = seed, maxIter = maxIter)
    val f2Vec = output.originStage.asInstanceOf[OpLDA]
    val testTransformedData = f2Vec.fit(inputDS).transform(inputDS)
    val estimate = testTransformedData.collect(output)
    val mse = computeMeanSqError(estimate, expected)
    val expectedMse = 0.5
    withClue(s"Computed mse $mse (expected $expectedMse)") {
      mse should be < expectedMse
    }
  }

  private def computeMeanSqError(estimate: Seq[OPVector], expected: Seq[Vector]): Double = {
    val n = estimate.length.toDouble
    estimate.zip(expected).map { case (est, exp) => Vectors.sqdist(est.value, exp) }.sum / n
  }
}

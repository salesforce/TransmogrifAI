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

package com.salesforce.op.evaluators

import com.salesforce.op.test.TestSparkContext
import org.apache.spark.mllib.evaluation.ExtendedBinaryClassificationMetrics
import org.junit.runner.RunWith
import org.scalatest.{Assertions, FlatSpec}
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class ExtendedBinaryClassificationMetricsTest extends FlatSpec with TestSparkContext {
  val numRecs = 400 // Number or records to use in threshold metrics tests
  val numBins = 100 // Number of bins for threshold metrics
  val scores = Seq.fill(numRecs)(math.random)
  val labels = Seq.fill(numRecs)(math.round(math.random).toDouble)

  val synthRDD = spark.sparkContext.parallelize(scores.zip(labels))

  Spec[ExtendedBinaryClassificationMetrics] should "produce deterministic threshold metrics" in {
    val numComparisons = 5

    for {i <- 1 to numComparisons} {
      val sparkMLMetrics = ExtendedBinaryClassificationMetrics(scoreAndLabels = synthRDD, numBins = numBins)
      sparkMLMetrics.confusionMatrixByThreshold().map(_._1).collect() should contain theSameElementsInOrderAs
        sparkMLMetrics.thresholds().collect()
    }
  }

}

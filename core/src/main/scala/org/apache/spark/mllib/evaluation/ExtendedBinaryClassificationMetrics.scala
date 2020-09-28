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

package org.apache.spark.mllib.evaluation

import java.lang.reflect.Method

import org.apache.spark.mllib.evaluation.binary.BinaryConfusionMatrix
import org.apache.spark.rdd.RDD

class ExtendedBinaryClassificationMetrics(
  override val scoreAndLabels: RDD[(Double, Double)],
  override val numBins: Int
) extends BinaryClassificationMetrics(scoreAndLabels, numBins) {

  def confusionMatrixByThreshold(): RDD[(Double, BinaryConfusionMatrix)] = {
    ExtendedBinaryClassificationMetrics.confusionMatrixByThreshold(this)
  }
}


object ExtendedBinaryClassificationMetrics {
  private lazy val confusionsLazyVal: Method = {
    val m = classOf[BinaryClassificationMetrics].getDeclaredMethod("confusions")
    m.setAccessible(true)
    m
  }

  def apply(
    scoreAndLabels: RDD[(Double, Double)],
    numBins: Int
  ): ExtendedBinaryClassificationMetrics = {
    new ExtendedBinaryClassificationMetrics(scoreAndLabels, numBins)
  }

  /**
   * Exposes the thresholded confusion matrices that Spark uses to calculate other derived metrics.
   * @return RDD of (threshold, BinaryConfusionMatrix) over all thresholds calculated
   */
  private def confusionMatrixByThreshold(bcm: BinaryClassificationMetrics) = {
    confusionsLazyVal.invoke(bcm).asInstanceOf[RDD[(Double, BinaryConfusionMatrix)]]
  }
}

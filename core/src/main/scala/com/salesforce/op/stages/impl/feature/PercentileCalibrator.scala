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

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.unary.{UnaryEstimator, UnaryModel}
import com.salesforce.op.utils.spark.RichMetadata._
import org.apache.spark.ml.feature.QuantileDiscretizer
import org.apache.spark.ml.param.IntParam
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types.MetadataBuilder

import scala.collection.Searching._


/**
 * Wraps around [[org.apache.spark.ml.feature.QuantileDiscretizer]]
 */
class PercentileCalibrator(uid: String = UID[PercentileCalibrator])
  extends UnaryEstimator[RealNN, RealNN](operationName = "percentCalibrator", uid = uid) {

  final val expectedNumBuckets = new IntParam(
    this, "expectedNumBuckets", "number of buckets to divide input data into"
  )
  setDefault(expectedNumBuckets, 100)

  def setExpectedNumBuckets(buckets: Int): this.type = set(expectedNumBuckets, buckets)

  def fitFn(dataset: Dataset[Option[Double]]): UnaryModel[RealNN, RealNN] = {

    val estimator: QuantileDiscretizer = new QuantileDiscretizer()
      .setNumBuckets($(expectedNumBuckets))
      .setRelativeError(0)
      .setInputCol(dataset.columns(0))

    val bucketizerModel = estimator.fit(dataset)

    val model = new PercentileCalibratorModel(
      splits = bucketizerModel.getSplits,
      actualNumBuckets = bucketizerModel.getSplits.length,
      expectedNumBuckets = $(expectedNumBuckets),
      operationName = operationName,
      uid = uid
    )

    val scaledBuckets = bucketizerModel.getSplits.map(v => model.transformFn(v.toRealNN).v.get)

    val meta = new MetadataBuilder()
      .putStringArray(PercentileCalibrator.OrigSplitsKey, bucketizerModel.getSplits.map(_.toString))
      .putStringArray(PercentileCalibrator.ScaledSplitsKey, scaledBuckets.map(_.toString)).build()
    setMetadata(meta.toSummaryMetadata())

    model
  }

}


final class PercentileCalibratorModel private[op]
(
  val splits: Array[Double],
  val actualNumBuckets: Int,
  val expectedNumBuckets: Int,
  operationName: String,
  uid: String
) extends UnaryModel[RealNN, RealNN](operationName = operationName, uid = uid) {

  def transformFn: RealNN => RealNN = (inScalar: RealNN) => {
    val calibrated = splits.search(inScalar.v.get) match {
      case Found(idx) => idx
      case InsertionPoint(idx) => idx
    }
    scale(actualNumBuckets, expectedNumBuckets, calibrated).toRealNN
  }

  private def scale(actualNumBuckets: Int, expectedBuckets: Int, calibrated: Int): Long = {
    if (actualNumBuckets >= expectedBuckets) {
      calibrated - 1 // make it start at zero
    } else {
      val (oldMin, newMin) = (0, 0)
      val (oldMax, newMax) = (Math.max(actualNumBuckets - 2, 0), Math.max(expectedBuckets - 1, 0))
      val oldRange = oldMax - oldMin
      oldRange match {
        case 0 => newMin
        case _ =>
          val newRange = (newMax - newMin).toDouble
          val newValue = (((calibrated - oldMin) * newRange) / oldRange) + newMin
          Math.min(newValue.round, newMax)
      }
    }
  }

}


case object PercentileCalibrator {
  val OrigSplitsKey: String = "origSplits"
  val ScaledSplitsKey: String = "scaledSplits"
}



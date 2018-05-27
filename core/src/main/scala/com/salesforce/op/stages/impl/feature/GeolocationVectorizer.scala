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
import com.salesforce.op.aggregators.{GeolocationFunctions, GeolocationMidpoint}
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.sequence.{SequenceEstimator, SequenceModel}
import com.salesforce.op.utils.spark.OpVectorMetadata
import com.salesforce.op.utils.spark.RichDataset._
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param.{BooleanParam, DoubleArrayParam}
import org.apache.spark.sql.{Dataset, Encoders}

/**
 * Converts a sequence of Geolocation features into a vector feature.
 * Can choose to fill null values with the mean or a constant
 *
 * @param uid uid for instance
 */
class GeolocationVectorizer
(
  uid: String = UID[GeolocationVectorizer],
  operationName: String = "vecGeo"
) extends SequenceEstimator[Geolocation, OPVector](operationName = operationName, uid = uid)
  with VectorizerDefaults with TrackNullsParam with GeolocationFunctions {

  private implicit val seqArrayEncoder = Encoders.kryo[Seq[Array[Double]]]

  final val fillValue = new DoubleArrayParam(this, "fillValue", "default value for FillWithConstant")
  setDefault(fillValue, TransmogrifierDefaults.DefaultGeolocation.toArray)

  final val withConstant = new BooleanParam(this, "fillWithConstant",
    "boolean to check if filling the nulls with a constant value")
  setDefault(withConstant, false)

  def setFillWithConstant(value: Geolocation): this.type = {
    set(fillValue, value.toArray)
    set(withConstant, true)
  }

  // Only supports filling with geographic centroid for now (same as in Geolocations aggregator)
  def setFillWithMean(): this.type = set(withConstant, false)

  private def constants(): Seq[Geolocation#Value] = {
    val size = getInputFeatures().length
    val defValue = $(fillValue).toSeq
    val constants = Seq.fill(size)(defValue)
    constants
  }

  private def means(dataset: Dataset[Seq[Geolocation#Value]]): Seq[Geolocation#Value] = {
    val preparedData: Dataset[Seq[Array[Double]]] =
      dataset.map(f => f.map(g => prepare(Geolocation(g))))
    val reducedData =
      if (preparedData.isEmpty) {
        Seq.empty[Array[Double]]
      } else {
        preparedData.reduce((a, b) =>
          a.zip(b).map { case (g1, g2) => GeolocationMidpoint.monoid.plus(g1, g2) }
        )
      }

    val means = reducedData.map(GeolocationMidpoint.present(_).value)
    means
  }

  /**
   * Compute the output vector metadata only from the input features. Vectorizers use this to derive
   * the full vector, including pivot columns or indicator features.
   *
   * @return Vector metadata from input features
   */
  private def vectorMetadata(withNullTracking: Boolean): OpVectorMetadata = {
    val tf = getTransientFeatures()
    val cols =
      if (withNullTracking) tf.flatMap(f => Seq.fill(3)(f.toColumnMetaData()) ++ Seq(f.toColumnMetaData(isNull = true)))
      else tf.flatMap(f => Seq.fill(3)(f.toColumnMetaData()))
    OpVectorMetadata(vectorOutputName, cols, Transmogrifier.inputFeaturesToHistory(tf, stageName))
  }

  override def vectorMetadataFromInputFeatures: OpVectorMetadata = vectorMetadata(withNullTracking = false)
  override def vectorMetadataWithNullIndicators: OpVectorMetadata = vectorMetadata(withNullTracking = true)

  def fitFn(dataset: Dataset[Seq[Geolocation#Value]]): SequenceModel[Geolocation, OPVector] = {
    if ($(trackNulls)) setMetadata(vectorMetadataWithNullIndicators.toMetadata)

    val fillValues = if ($(withConstant)) constants() else means(dataset)

    new GeolocationVectorizerModel(
      fillValues = fillValues, trackNulls = $(trackNulls), operationName = operationName, uid = uid)
  }

}

final class GeolocationVectorizerModel private[op]
(
  val fillValues: Seq[Seq[Double]],
  val trackNulls: Boolean,
  operationName: String,
  uid: String
) extends SequenceModel[Geolocation, OPVector](operationName = operationName, uid = uid)
  with VectorizerDefaults {

  private val RepresentationOfEmpty: Seq[Double] = Array(0.0, 0.0, 0.0)

  def transformFn: Seq[Geolocation] => OPVector = row => {
    val replaced =
      if (!trackNulls) {
        row.zip(fillValues).flatMap { case (r, m) => if (r.isEmpty) m else r.value }
      }
      else {
        row.zip(fillValues).flatMap { case (r, m) =>
          val meanToUse: Seq[Double] = if (m.isEmpty) RepresentationOfEmpty else m
          // Unlike the RealVectorizer case, we need parents here since r.value is a list and :+ takes precedence
          (if (r.isEmpty) meanToUse else r.value) :+ (r.isEmpty: Double)
        }
      }
    Vectors.dense(replaced.toArray).toOPVector
  }
}

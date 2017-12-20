/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.UID
import com.salesforce.op.aggregators.{GeolocationFunctions, Geolocations}
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.sequence.{SequenceEstimator, SequenceModel}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param.{BooleanParam, DoubleArrayParam}
import org.apache.spark.sql.{Dataset, Encoders}
import com.salesforce.op.utils.spark.RichDataset._

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
          a.zip(b).map { case (g1, g2) => Geolocations.monoid.plus(g1, g2) }
        )
      }

    val means = reducedData.map(Geolocations.present(_).value)
    means
  }

  def fitFn(dataset: Dataset[Seq[Geolocation#Value]]): SequenceModel[Geolocation, OPVector] = {
    if ($(trackNulls)) setMetadata(vectorMetadataWithNullIndicators.toMetadata)

    val fillValues = if ($(withConstant)) constants() else means(dataset)

    new GeolocationVectorizerModel(
      fillValues = fillValues, trackNulls = $(trackNulls), operationName = operationName, uid = uid)
  }

}

private final class GeolocationVectorizerModel
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
        row.zip(fillValues).
          flatMap { case (r, m) => if (r.isEmpty) m else r.value }
      }
      else {
        row.zip(fillValues).
          flatMap { case (r, m) =>
            val meanToUse: Seq[Double] = if (m.isEmpty) RepresentationOfEmpty else m
            // Unlike the RealVectorizer case, we need parents here since r.value is a list and :+ takes precedence
            (if (r.isEmpty) meanToUse else r.value) :+ booleanToDouble(r.isEmpty)
          }
      }

    Vectors.dense(replaced.toArray).toOPVector
  }
}
